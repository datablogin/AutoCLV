"""Data loading utilities for MCP server."""

import json
from datetime import datetime
from pathlib import Path

from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.instance import mcp
from analytics.services.mcp_server.state import get_shared_state


class LoadTransactionsRequest(BaseModel):
    """Request to load transaction data from file."""

    file_path: str = Field(
        default="synthetic_transactions.json",
        description="Path to transaction data file (relative to project root or absolute path)",
    )
    limit: int | None = Field(
        default=None,
        description="Optional limit on number of transactions to load",
    )


class LoadTransactionsResponse(BaseModel):
    """Response containing loaded transaction data summary."""

    total_count: int
    loaded_count: int
    file_path: str
    message: str
    date_range: tuple[str, str] | None = None
    customer_count: int | None = None
    sample_transactions: list[dict] | None = None


@mcp.tool()
async def load_transactions(
    request: LoadTransactionsRequest, ctx: Context
) -> LoadTransactionsResponse:
    """Load transaction data from a JSON file.

    This tool loads transaction data from the specified file path. If the path is
    relative, it will be resolved relative to the project root directory.

    Args:
        request: Request containing file path and optional limit
        ctx: MCP context

    Returns:
        Response containing transaction data and metadata

    Examples:
        Load all synthetic transactions:
            load_transactions(file_path="synthetic_transactions.json")

        Load first 1000 transactions:
            load_transactions(file_path="synthetic_transactions.json", limit=1000)
    """
    # Resolve file path
    file_path = Path(request.file_path)

    # If relative path, resolve from project root
    if not file_path.is_absolute():
        # Project root is 5 levels up from this file
        # analytics/services/mcp_server/tools/data_loader.py -> project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        file_path = project_root / file_path
    else:
        # For absolute paths, use project root as allowed base
        project_root = Path(__file__).parent.parent.parent.parent.parent

    # Resolve to absolute path and validate it stays within project directory
    resolved_path = file_path.resolve()
    try:
        resolved_path.relative_to(project_root.resolve())
    except ValueError as e:
        raise ValueError(
            f"Path {resolved_path} is outside allowed directory {project_root.resolve()}. "
            f"Only files within the project directory can be loaded."
        ) from e

    # Use validated path
    file_path = resolved_path

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {file_path}\n"
            f"Tried absolute path: {file_path.absolute()}"
        )

    # Load transaction data
    try:
        with open(file_path) as f:
            transactions = json.load(f)

        if not isinstance(transactions, list):
            raise ValueError(
                f"Expected transaction data to be a list, got {type(transactions)}"
            )

        total_count = len(transactions)

        # Apply limit if specified
        if request.limit is not None and request.limit > 0:
            transactions = transactions[: request.limit]
            message = (
                f"Loaded {len(transactions)} of {total_count} transactions "
                f"from {file_path.name}"
            )
        else:
            message = f"Loaded {total_count} transactions from {file_path.name}"

        # Parse datetime fields (required for data mart builder)
        parsed_transactions = []
        for idx, item in enumerate(transactions):
            txn = dict(item)

            # Parse order_ts or event_ts if present
            for ts_field in ["order_ts", "event_ts"]:
                if ts_field in txn:
                    value = txn[ts_field]

                    # Parse string to datetime
                    if isinstance(value, str):
                        try:
                            txn[ts_field] = datetime.fromisoformat(value)
                        except (ValueError, TypeError) as e:
                            raise ValueError(
                                f"Transaction {idx}: Failed to parse {ts_field} as datetime: {value}"
                            ) from e

                    # Validate non-string values are datetime objects
                    elif not isinstance(value, datetime):
                        raise TypeError(
                            f"Transaction {idx}: {ts_field} must be a datetime or ISO format string, "
                            f"got {type(value).__name__}"
                        )

            parsed_transactions.append(txn)

        transactions = parsed_transactions

        # Store transactions in shared state for use by orchestration and other tools
        shared_state = get_shared_state()
        shared_state.set("transactions", transactions)

        # Calculate summary statistics
        unique_customers = len(
            set(t.get("customer_id") for t in transactions if t.get("customer_id"))
        )

        # Get date range (now working with datetime objects)
        dates = []
        for t in transactions:
            ts = t.get("event_ts") or t.get("order_ts")
            if ts and isinstance(ts, datetime):
                dates.append(ts)

        date_range = None
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range = (min_date.isoformat(), max_date.isoformat())

        # Store metadata
        shared_state.set(
            "transactions_metadata",
            {
                "total_count": total_count,
                "loaded_count": len(transactions),
                "file_path": str(file_path),
                "customer_count": unique_customers,
                "date_range": date_range,
            },
        )

        # Create sample with datetime objects converted to strings for JSON serialization
        sample_transactions = None
        if transactions:
            sample_transactions = []
            for txn in transactions[:5]:
                sample_txn = dict(txn)
                # Convert datetime objects to ISO strings for JSON serialization
                for key, value in sample_txn.items():
                    if isinstance(value, datetime):
                        sample_txn[key] = value.isoformat()
                sample_transactions.append(sample_txn)

        return LoadTransactionsResponse(
            total_count=total_count,
            loaded_count=len(transactions),
            file_path=str(file_path),
            message=message,
            customer_count=unique_customers,
            date_range=date_range,
            sample_transactions=sample_transactions,
        )

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in transaction file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading transaction data: {e}")

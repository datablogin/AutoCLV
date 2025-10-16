#!/bin/bash

# AutoCLV — Local PR Review Script (Claude Code)
# Adapted for AutoCLV project with analytics backend integration
# Dependencies: gh (GitHub CLI), claude (Claude Code CLI), jq

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Config
FOCUS_AREAS=""         # comma-separated; supported: security,performance,testing,style,analytics,ml,streaming,clv,data-quality,docs
MODEL=""               # optional model override if supported by claude CLI
POST_COMMENT=true
OUTPUT_MODE="comment"   # comment|draft-comment|file
DRY_RUN=false
MAX_DIFF_LINES=500
CHUNK_FILES=0  # 0 = review all at once; N = review N files per chunk

ORIGINAL_BRANCH=$(git branch --show-current || true)

usage() {
  echo "Usage: $0 [OPTIONS] [PR_NUMBER]"
  echo ""
  echo "Options:"
  echo "  --focus AREA        Focus review: security, performance, testing, style, analytics, ml, streaming, clv, data-quality, docs"
  echo "  --model MODEL       Use specific Claude model (if supported by CLI)"
  echo "  --save-file         Save review to file instead of posting a comment"
  echo "  --draft-comment     Post review as a draft PR comment"
  echo "  --max-diff-lines N  Maximum diff lines to include (default: 500, 0 = no limit)"
  echo "  --chunk-files N     Review N files per chunk (default: 0 = all at once)"
  echo "  --dry-run           Show what would be reviewed without calling Claude"
  echo "  --help              Show this help"
  echo ""
  echo "Examples:"
  echo "  $0                             # Review current PR and post as comment"
  echo "  $0 12                          # Review PR #12 and post as comment"
  echo "  $0 --focus analytics,ml 12     # Focus analytics and ML components"
  echo "  $0 --chunk-files 10 12         # Review PR #12 in chunks of 10 files each"
  echo "  $0 --chunk-files 5 --max-diff-lines 0 12  # Review 5 files at a time, no line limit"
}

check_dependencies() {
  local missing=()
  command -v gh >/dev/null 2>&1 || missing+=("GitHub CLI (gh)")
  command -v claude >/dev/null 2>&1 || missing+=("Claude Code CLI (claude)")
  command -v jq >/dev/null 2>&1 || missing+=("jq")
  if [ ${#missing[@]} -ne 0 ]; then
    echo -e "${RED}Missing dependencies:${NC} ${missing[*]}"
    exit 1
  fi
}

check_dependencies

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --focus) FOCUS_AREAS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --save-file) POST_COMMENT=false; OUTPUT_MODE="file"; shift ;;
    --draft-comment) POST_COMMENT=true; OUTPUT_MODE="draft-comment"; shift ;;
    --max-diff-lines) MAX_DIFF_LINES="$2"; shift 2 ;;
    --chunk-files) CHUNK_FILES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help) usage; exit 0 ;;
    -*) echo -e "${RED}Unknown option: $1${NC}"; usage; exit 1 ;;
    *)
      if [[ $1 =~ ^[0-9]+$ ]]; then PR_NUM=$1; shift; else echo -e "${RED}Invalid PR number: $1${NC}"; usage; exit 1; fi ;;
  esac
done

# Resolve PR number
if [ -z "$PR_NUM" ]; then
  PR_NUM=$(gh pr view --json number -q .number 2>/dev/null || echo "")
  if [ -z "$PR_NUM" ]; then
    echo -e "${RED}Not on a PR branch; specify PR number${NC}"; usage; exit 1
  fi
fi

gh pr view "$PR_NUM" >/dev/null || { echo -e "${RED}PR #$PR_NUM not found${NC}"; exit 1; }

# Heuristics for additional prompts
has_analytics_files() { gh pr diff "$PR_NUM" --name-only | grep -E "analytics/libs/(analytics_core|data_processing|streaming_analytics)" >/dev/null 2>&1; }
has_ml_files()        { gh pr diff "$PR_NUM" --name-only | grep -E "analytics/libs/ml_models|analytics/services/ml_inference" >/dev/null 2>&1; }
has_streaming_files() { gh pr diff "$PR_NUM" --name-only | grep -E "analytics/libs/streaming_analytics|analytics/services/data_ingestion" >/dev/null 2>&1; }
has_clv_files()       { gh pr diff "$PR_NUM" --name-only | grep -E "customer_base_audit|clv|lifetime_value" >/dev/null 2>&1; }
has_api_files()       { gh pr diff "$PR_NUM" --name-only | grep -E "analytics/services/analytics_api|analytics/libs/api_common" >/dev/null 2>&1; }

create_diff_summary() {
  local pr_num="$1"; local max_lines="$2"; shift 2
  local files=("$@")
  local full
  if [ ${#files[@]} -eq 0 ]; then
    full=$(gh pr diff "$pr_num")
  else
    full=$(gh pr diff "$pr_num" -- "${files[@]}")
  fi
  if [ "$max_lines" -eq 0 ]; then echo "$full"; return; fi
  local n; n=$(echo "$full" | wc -l | tr -d ' ')
  if [ "$n" -le "$max_lines" ]; then echo "$full"; else
    echo "### ⚠️ Large Diff Summary (${n} lines total, showing first ${max_lines} lines)"
    echo ""
    echo "\`\`\`diff"; echo "$full" | head -n "$max_lines"; echo "\`\`\`"
    local owner; owner=$(gh repo view --json owner -q '.owner.login')
    local name;  name=$(gh repo view --json name -q '.name')
    echo "Full diff: https://github.com/${owner}/${name}/pull/${pr_num}/files"
  fi
}

generate_review_prompt() {
  local base="Please review this pull request for:\n- Code quality and correctness\n- Potential bugs\n- Performance considerations\n- Security concerns\n- Test coverage and determinism\n\nBe constructive and specific."
  local domain=""
  if has_analytics_files || [[ "$FOCUS_AREAS" == *"analytics"* ]]; then
    domain+="\n\nFor analytics components:\n- Data processing pipeline correctness and efficiency\n- Database schema and migration safety\n- Async/await patterns and error handling\n- Data quality validation and lineage tracking\n- Observability and monitoring integration"
  fi
  if has_ml_files || [[ "$FOCUS_AREAS" == *"ml"* ]]; then
    domain+="\n\nFor ML components:\n- Model serving and inference pipeline safety\n- MLflow integration and experiment tracking\n- Feature store consistency and versioning\n- Model monitoring and drift detection\n- Performance optimization for real-time inference"
  fi
  if has_streaming_files || [[ "$FOCUS_AREAS" == *"streaming"* ]]; then
    domain+="\n\nFor streaming analytics:\n- Kafka producer/consumer patterns and error handling\n- Stream processing windowing and aggregation correctness\n- Real-time ML pipeline latency and throughput\n- WebSocket connection management and authentication\n- Auto-scaling and monitoring integration"
  fi
  if has_clv_files || [[ "$FOCUS_AREAS" == *"clv"* ]]; then
    domain+="\n\nFor CLV components:\n- Customer lifetime value calculation accuracy\n- Statistical model correctness and validation\n- Data input validation and edge case handling\n- Performance for large customer datasets\n- Integration with analytics backend services"
  fi
  if has_api_files || [[ "$FOCUS_AREAS" == *"api"* ]]; then
    domain+="\n\nFor API services:\n- FastAPI endpoint design and validation\n- Pydantic model correctness and serialization\n- Authentication and authorization patterns\n- Rate limiting and DOS protection\n- OpenAPI documentation completeness"
  fi
  case "$FOCUS_AREAS" in
    *security*) domain+="\n\nFocus security: input validation, schema validation, SSRF/download risks, secrets handling.";;
  esac
  case "$FOCUS_AREAS" in
    *performance*) domain+="\n\nFocus performance: analytics pipeline throughput, ML inference latency, streaming processing speed, database query optimization.";;
  esac
  case "$FOCUS_AREAS" in
    *testing*) domain+="\n\nFocus testing: analytics pipeline testing, ML model validation, streaming integration tests, async test patterns.";;
  esac
  case "$FOCUS_AREAS" in
    *data-quality*) domain+="\n\nFocus data quality: validation framework correctness, profiling accuracy, lineage tracking, alerting mechanisms.";;
  esac
  case "$FOCUS_AREAS" in
    *style*) domain+="\n\nFocus style: readability, naming, modularity, docs.";;
  esac
  case "$FOCUS_AREAS" in
    *docs*) domain+="\n\nFocus docs: README/architecture clarity, analytics component documentation, API documentation, integration examples.";;
  esac
  echo -e "$base$domain"
}

# PR info
PR_INFO=$(gh pr view "$PR_NUM" --json title,author,baseRefName,headRefName,additions,deletions,changedFiles,commits)
PR_TITLE=$(echo "$PR_INFO" | jq -r .title)
PR_AUTHOR=$(echo "$PR_INFO" | jq -r .author.login)
PR_BRANCH=$(echo "$PR_INFO" | jq -r .headRefName)
PR_BASE_BRANCH=$(echo "$PR_INFO" | jq -r .baseRefName)
PR_ADDITIONS=$(echo "$PR_INFO" | jq -r .additions)
PR_DELETIONS=$(echo "$PR_INFO" | jq -r .deletions)
PR_CHANGED_FILES=$(echo "$PR_INFO" | jq -r .changedFiles)
PR_COMMITS=$(echo "$PR_INFO" | jq -r '.commits | length')

echo -e "${GREEN}Reviewing PR #$PR_NUM: $PR_TITLE${NC}"
echo -e "Author: $PR_AUTHOR"
echo -e "Branch: $PR_BRANCH → $PR_BASE_BRANCH"
echo -e "Changes: ${GREEN}+$PR_ADDITIONS${NC} ${RED}-$PR_DELETIONS${NC} lines across $PR_CHANGED_FILES files"
echo -e "Commits: $PR_COMMITS"
if [ -n "$FOCUS_AREAS" ]; then echo -e "Focus: ${BLUE}$FOCUS_AREAS${NC}"; fi

echo ""

# Ensure on PR branch
CURRENT_BRANCH=$(git branch --show-current || true)
if [ "$CURRENT_BRANCH" != "$PR_BRANCH" ]; then
  echo -e "${YELLOW}Checking out PR branch...${NC}"
  gh pr checkout "$PR_NUM"
fi

REVIEW_PROMPT=$(generate_review_prompt)

# Get all changed files
ALL_FILES=($(gh pr diff "$PR_NUM" --name-only))
TOTAL_FILES=${#ALL_FILES[@]}

# Determine chunking strategy
if [ "$CHUNK_FILES" -gt 0 ] && [ "$TOTAL_FILES" -gt "$CHUNK_FILES" ]; then
  NUM_CHUNKS=$(( (TOTAL_FILES + CHUNK_FILES - 1) / CHUNK_FILES ))
  echo -e "${YELLOW}Large PR detected: splitting into $NUM_CHUNKS chunks ($CHUNK_FILES files each)${NC}"
else
  NUM_CHUNKS=1
  CHUNK_FILES=$TOTAL_FILES
fi

echo -e "${BLUE}Preparing PR context (max diff lines: $MAX_DIFF_LINES)...${NC}"

# Function to create PR context for specific files
create_pr_context() {
  local chunk_num="$1"; local chunk_files=("${@:2}")
  cat <<EOF
### PR Context
- **Title:** $PR_TITLE
- **Author:** $PR_AUTHOR
- **Branch:** $PR_BRANCH → $PR_BASE_BRANCH
- **Additions:** $PR_ADDITIONS
- **Deletions:** $PR_DELETIONS
- **Files Changed:** $PR_CHANGED_FILES
- **Commits:** $PR_COMMITS
$([ "$NUM_CHUNKS" -gt 1 ] && echo "- **Review Chunk:** $chunk_num of $NUM_CHUNKS")

### Files in this chunk:
\`\`\`
$(printf '%s\n' "${chunk_files[@]}")
\`\`\`

### Code Changes:
$(create_diff_summary "$PR_NUM" "$MAX_DIFF_LINES" "${chunk_files[@]}")
EOF
}

# For single chunk, use original variable name for backwards compatibility
if [ "$NUM_CHUNKS" -eq 1 ]; then
  PR_CONTEXT=$(create_pr_context 1 "${ALL_FILES[@]}")
fi

if [ "$DRY_RUN" = true ]; then
  echo -e "${BLUE}DRY RUN — Files to review:${NC}"
  gh pr diff "$PR_NUM" --name-only | sed 's/^/  - /'
  echo ""; echo "Generated prompt:"; echo "$REVIEW_PROMPT" | sed 's/^/  /'
  exit 0
fi

case "$OUTPUT_MODE" in
  comment|draft-comment)
    echo -e "${YELLOW}Running Claude review and posting to PR...${NC}"
    REVIEWS_DIR=$(mktemp -d)
    ALL_REVIEWS=()

    # Review each chunk
    for (( chunk=0; chunk<NUM_CHUNKS; chunk++ )); do
      START_IDX=$((chunk * CHUNK_FILES))
      END_IDX=$((START_IDX + CHUNK_FILES))
      [ "$END_IDX" -gt "$TOTAL_FILES" ] && END_IDX=$TOTAL_FILES
      CHUNK_FILES_ARRAY=("${ALL_FILES[@]:$START_IDX:$((END_IDX - START_IDX))}")
      CHUNK_NUM=$((chunk + 1))

      if [ "$NUM_CHUNKS" -gt 1 ]; then
        echo -e "${BLUE}Reviewing chunk $CHUNK_NUM/$NUM_CHUNKS (files $((START_IDX + 1))-$END_IDX)...${NC}"
      fi

      CHUNK_CONTEXT=$(create_pr_context "$CHUNK_NUM" "${CHUNK_FILES_ARRAY[@]}")
      TMP=$(mktemp)
      echo "$CHUNK_CONTEXT" > "$TMP"
      echo -e "\n---\n\n$REVIEW_PROMPT" >> "$TMP"
      OUT="${REVIEWS_DIR}/chunk_${CHUNK_NUM}.md"

      if claude chat < "$TMP" > "$OUT" 2>&1; then
        ALL_REVIEWS+=("$OUT")
        if [ "$NUM_CHUNKS" -gt 1 ]; then
          echo -e "${GREEN}✓ Chunk $CHUNK_NUM reviewed${NC}"
        fi
      else
        echo -e "${RED}✗ Chunk $CHUNK_NUM review failed${NC}"
        [ -f "$OUT" ] && head -n 20 "$OUT" || true
      fi
      rm -f "$TMP"
    done

    # Aggregate reviews
    if [ ${#ALL_REVIEWS[@]} -gt 0 ]; then
      COMMENT=$(mktemp)
      cat > "$COMMENT" <<EOC
# 🔍 Claude Code Review

## Review Feedback

EOC
      if [ "$NUM_CHUNKS" -gt 1 ]; then
        for (( i=0; i<${#ALL_REVIEWS[@]}; i++ )); do
          echo "### Chunk $((i + 1)) of ${#ALL_REVIEWS[@]}" >> "$COMMENT"
          echo "" >> "$COMMENT"
          cat "${ALL_REVIEWS[$i]}" >> "$COMMENT"
          echo "" >> "$COMMENT"
          echo "---" >> "$COMMENT"
          echo "" >> "$COMMENT"
        done
      else
        cat "${ALL_REVIEWS[0]}" >> "$COMMENT"
      fi
      cat >> "$COMMENT" <<EOC

---
*Generated by AutoCLV PR Review Tool*
EOC
      if [ "$OUTPUT_MODE" = "draft-comment" ]; then
        gh pr comment "$PR_NUM" --body-file "$COMMENT" --draft >/dev/null || true
      else
        gh pr comment "$PR_NUM" --body-file "$COMMENT" >/dev/null || true
      fi
      echo -e "${GREEN}✓ Review comment posted${NC}"
      rm -f "$COMMENT"
    else
      echo -e "${RED}✗ All reviews failed${NC}"
    fi
    rm -rf "$REVIEWS_DIR"
    ;;
  file)
    DATE=$(date +%Y%m%d_%H%M)
    OUTDIR="reviews/autoclv"
    mkdir -p "$OUTDIR"
    SUF=""; [ -n "$FOCUS_AREAS" ] && SUF="-$(echo "$FOCUS_AREAS" | tr ',' '-')"
    OUTFILE="$OUTDIR/pr-${PR_NUM}${SUF}-${DATE}.md"
    echo -e "${YELLOW}Running Claude review and saving to $OUTFILE...${NC}"
    {
      echo "# 🔍 Claude Code Review: PR #$PR_NUM"; echo "";
      echo "**Title:** $PR_TITLE  "; echo "**Author:** $PR_AUTHOR  ";
      echo "**Date:** $(date +"%Y-%m-%d %H:%M:%S")  ";
      echo "**Branch:** $PR_BRANCH → $PR_BASE_BRANCH"; echo "";
      [ "$NUM_CHUNKS" -gt 1 ] && echo "**Review Mode:** Chunked ($NUM_CHUNKS chunks, $CHUNK_FILES files each)"; echo "";
    } > "$OUTFILE"

    # Review each chunk
    for (( chunk=0; chunk<NUM_CHUNKS; chunk++ )); do
      START_IDX=$((chunk * CHUNK_FILES))
      END_IDX=$((START_IDX + CHUNK_FILES))
      [ "$END_IDX" -gt "$TOTAL_FILES" ] && END_IDX=$TOTAL_FILES
      CHUNK_FILES_ARRAY=("${ALL_FILES[@]:$START_IDX:$((END_IDX - START_IDX))}")
      CHUNK_NUM=$((chunk + 1))

      if [ "$NUM_CHUNKS" -gt 1 ]; then
        echo -e "${BLUE}Reviewing chunk $CHUNK_NUM/$NUM_CHUNKS (files $((START_IDX + 1))-$END_IDX)...${NC}"
        echo "" >> "$OUTFILE"
        echo "## Chunk $CHUNK_NUM of $NUM_CHUNKS" >> "$OUTFILE"
        echo "" >> "$OUTFILE"
      fi

      CHUNK_CONTEXT=$(create_pr_context "$CHUNK_NUM" "${CHUNK_FILES_ARRAY[@]}")
      echo "$CHUNK_CONTEXT" >> "$OUTFILE"
      echo "" >> "$OUTFILE"
      echo "---" >> "$OUTFILE"
      echo "" >> "$OUTFILE"
      echo "### Review Prompt Used" >> "$OUTFILE"
      echo "" >> "$OUTFILE"
      echo "$REVIEW_PROMPT" >> "$OUTFILE"
      echo "" >> "$OUTFILE"
      echo "---" >> "$OUTFILE"
      echo "" >> "$OUTFILE"
      echo "### Claude Review Output" >> "$OUTFILE"
      echo "" >> "$OUTFILE"

      TMP=$(mktemp)
      echo "$CHUNK_CONTEXT" > "$TMP"
      echo -e "\n---\n\n$REVIEW_PROMPT" >> "$TMP"

      if claude chat < "$TMP" >> "$OUTFILE" 2>&1; then
        if [ "$NUM_CHUNKS" -gt 1 ]; then
          echo -e "${GREEN}✓ Chunk $CHUNK_NUM reviewed${NC}"
        fi
      else
        echo -e "${RED}✗ Chunk $CHUNK_NUM review failed${NC}"
      fi
      rm -f "$TMP"

      if [ "$NUM_CHUNKS" -gt 1 ] && [ "$CHUNK_NUM" -lt "$NUM_CHUNKS" ]; then
        echo "" >> "$OUTFILE"
        echo "---" >> "$OUTFILE"
        echo "" >> "$OUTFILE"
      fi
    done

    echo -e "${GREEN}✓ Review saved: $OUTFILE${NC}"
    ;;
esac

# Return to original branch
if [ -n "$ORIGINAL_BRANCH" ] && [ "$ORIGINAL_BRANCH" != "$PR_BRANCH" ]; then
  echo -e "${YELLOW}Returning to branch: $ORIGINAL_BRANCH${NC}"
  git checkout "$ORIGINAL_BRANCH"
fi

echo -e "${BLUE}AutoCLV PR Review Script ready.${NC}"


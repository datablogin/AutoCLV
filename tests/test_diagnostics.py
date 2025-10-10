"""Tests for model diagnostics module.

Tests MCMC convergence checks, posterior predictive checks, and trace plot generation.
"""

import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest

from customer_base_audit.validation.diagnostics import (
    ConvergenceDiagnostics,
    PosteriorPredictiveStats,
    check_mcmc_convergence,
    plot_trace_diagnostics,
    posterior_predictive_check,
)


class TestCheckMCMCConvergence:
    """Tests for check_mcmc_convergence()."""

    def test_convergence_check_with_good_chains(self):
        """Test convergence check passes for well-converged chains."""
        # Create mock InferenceData with good R-hat values
        # Simulate 4 chains, 1000 draws each, 3 parameters
        np.random.seed(42)
        posterior = {
            "param_1": np.random.normal(0, 1, size=(4, 1000)),
            "param_2": np.random.normal(5, 2, size=(4, 1000)),
            "param_3": np.random.normal(-1, 0.5, size=(4, 1000)),
        }
        idata = az.from_dict(posterior)

        # Check convergence
        result = check_mcmc_convergence(idata, r_hat_threshold=1.1)

        assert isinstance(result, ConvergenceDiagnostics)
        assert result.converged is True
        assert result.max_r_hat < 1.1
        assert len(result.failed_parameters) == 0
        assert result.r_hat_threshold == 1.1
        assert result.ess_threshold == 400.0
        assert result.min_ess_bulk > 0
        assert result.min_ess_tail > 0
        assert isinstance(result.summary, pd.DataFrame)
        assert "r_hat" in result.summary.columns

    def test_convergence_check_with_divergent_chains(self):
        """Test convergence check fails for poorly-converged chains."""
        # Create mock InferenceData with one poorly-mixed chain
        np.random.seed(42)

        # Chain 1-3: centered around 0
        # Chain 4: centered around 10 (poorly mixed)
        param_1 = np.concatenate(
            [
                np.random.normal(0, 1, size=(3, 1000)),
                np.random.normal(10, 1, size=(1, 1000)),
            ]
        )

        posterior = {"param_1": param_1}
        idata = az.from_dict(posterior)

        # Check convergence
        result = check_mcmc_convergence(idata, r_hat_threshold=1.1)

        assert result.converged is False
        assert result.max_r_hat > 1.1
        assert len(result.failed_parameters) > 0
        assert "param_1" in result.failed_parameters

    def test_convergence_check_custom_threshold(self):
        """Test convergence check with custom R-hat threshold."""
        np.random.seed(42)
        posterior = {"param_1": np.random.normal(0, 1, size=(4, 1000))}
        idata = az.from_dict(posterior)

        # Use very strict threshold
        result = check_mcmc_convergence(idata, r_hat_threshold=1.01)

        assert result.r_hat_threshold == 1.01
        # May or may not converge with this strict threshold
        # Just check the result is valid
        assert isinstance(result.converged, bool)

    def test_convergence_check_with_single_chain(self):
        """Test convergence check with single chain (R-hat will be NaN)."""
        # Create InferenceData with single chain
        # ArviZ will compute R-hat but it will be NaN or give warnings
        np.random.seed(42)
        posterior = {"param_1": np.random.normal(0, 1, size=(1, 1000))}
        idata = az.from_dict(posterior)

        # Should still work but may have NaN R-hat values
        result = check_mcmc_convergence(idata)
        assert isinstance(result, ConvergenceDiagnostics)
        # Single chain R-hat may be NaN, which is > 1.1 comparison will be False
        # Just check we got a result
        assert isinstance(result.converged, bool)


class TestPosteriorPredictiveCheck:
    """Tests for posterior_predictive_check()."""

    def test_ppc_with_perfect_predictions(self):
        """Test PPC when predictions match observations perfectly."""
        np.random.seed(42)
        observed = pd.Series([2.0, 5.0, 1.0, 3.0, 4.0])

        # Create posterior samples centered on observed values
        posterior_samples = observed.values.reshape(1, -1) + np.random.normal(
            0, 0.01, size=(1000, 5)
        )

        stats = posterior_predictive_check(observed, posterior_samples)

        assert isinstance(stats, PosteriorPredictiveStats)
        assert stats.observed_mean == pytest.approx(3.0, abs=0.1)
        assert stats.predicted_mean == pytest.approx(3.0, abs=0.1)
        assert stats.coverage_95 > 0.9  # Most observations within 95% CI
        assert stats.median_abs_error < 0.5  # Small error

    def test_ppc_with_biased_predictions(self):
        """Test PPC when predictions are systematically biased."""
        np.random.seed(42)
        observed = pd.Series([2.0, 5.0, 1.0, 3.0, 4.0])

        # Create posterior samples with systematic bias (overestimate by 2)
        posterior_samples = (observed.values + 2.0).reshape(1, -1) + np.random.normal(
            0, 0.1, size=(1000, 5)
        )

        stats = posterior_predictive_check(observed, posterior_samples)

        assert stats.observed_mean == pytest.approx(3.0, abs=0.1)
        assert stats.predicted_mean == pytest.approx(5.0, abs=0.1)  # Biased upward
        assert stats.coverage_95 == 0.0  # No observations within CI (all too low)
        assert stats.median_abs_error > 1.5  # Large systematic error

    def test_ppc_with_point_predictions(self):
        """Test PPC with 1D point predictions instead of full posterior."""
        np.random.seed(42)
        observed = pd.Series([2.0, 5.0, 1.0, 3.0, 4.0])
        point_predictions = np.array([2.1, 4.9, 1.1, 3.0, 3.9])

        # Should handle 1D array
        stats = posterior_predictive_check(observed, point_predictions)

        assert isinstance(stats, PosteriorPredictiveStats)
        assert stats.observed_mean == pytest.approx(3.0, abs=0.1)
        assert stats.median_abs_error < 0.2

    def test_ppc_statistics_ranges(self):
        """Test PPC statistics are in valid ranges."""
        np.random.seed(42)
        observed = pd.Series([2.0, 5.0, 1.0, 3.0, 4.0])
        posterior_samples = np.random.normal(3, 1, size=(1000, 5))

        stats = posterior_predictive_check(observed, posterior_samples)

        # Coverage should be between 0 and 1
        assert 0.0 <= stats.coverage_95 <= 1.0

        # MAE should be non-negative
        assert stats.median_abs_error >= 0.0

        # Standard deviations should be positive
        assert stats.observed_std > 0.0
        assert stats.predicted_std > 0.0

    def test_ppc_empty_data_raises_error(self):
        """Test PPC raises error for empty observed data."""
        observed = pd.Series([])
        posterior_samples = np.array([[1, 2, 3]])

        with pytest.raises(ValueError, match="observed_data cannot be empty"):
            posterior_predictive_check(observed, posterior_samples)

    def test_ppc_shape_mismatch_raises_error(self):
        """Test PPC raises error when dimensions don't match."""
        observed = pd.Series([1.0, 2.0, 3.0])
        posterior_samples = np.random.normal(0, 1, size=(1000, 5))  # 5 != 3

        with pytest.raises(ValueError, match="Shape mismatch"):
            posterior_predictive_check(observed, posterior_samples)


class TestPlotTraceDiagnostics:
    """Tests for plot_trace_diagnostics()."""

    def test_trace_plot_generation(self):
        """Test trace plot is generated and saved."""
        # Create mock InferenceData
        np.random.seed(42)
        posterior = {
            "param_1": np.random.normal(0, 1, size=(4, 1000)),
            "param_2": np.random.normal(5, 2, size=(4, 1000)),
        }
        idata = az.from_dict(posterior)

        # Use temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trace_plot.png"

            # Generate plot
            plot_trace_diagnostics(idata, str(output_path))

            # Verify file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # Non-empty file

    def test_trace_plot_creates_output_directory(self):
        """Test trace plot creates output directory if it doesn't exist."""
        np.random.seed(42)
        posterior = {"param_1": np.random.normal(0, 1, size=(4, 1000))}
        idata = az.from_dict(posterior)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested directory that doesn't exist yet
            output_path = Path(tmpdir) / "diagnostics" / "subdir" / "trace_plot.png"

            plot_trace_diagnostics(idata, str(output_path))

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_trace_plot_with_var_names(self):
        """Test trace plot with specific variable selection."""
        np.random.seed(42)
        posterior = {
            "param_1": np.random.normal(0, 1, size=(4, 1000)),
            "param_2": np.random.normal(5, 2, size=(4, 1000)),
            "param_3": np.random.normal(-1, 0.5, size=(4, 1000)),
        }
        idata = az.from_dict(posterior)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trace_plot.png"

            # Plot only param_1 and param_2
            plot_trace_diagnostics(
                idata, str(output_path), var_names=["param_1", "param_2"]
            )

            assert output_path.exists()

    def test_trace_plot_custom_figsize(self):
        """Test trace plot with custom figure size."""
        np.random.seed(42)
        posterior = {"param_1": np.random.normal(0, 1, size=(4, 1000))}
        idata = az.from_dict(posterior)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trace_plot.png"

            # Use custom figure size
            plot_trace_diagnostics(idata, str(output_path), figsize=(16, 10))

            assert output_path.exists()


class TestIntegrationWithRealModels:
    """Integration tests with actual BG/NBD and Gamma-Gamma models."""

    @pytest.mark.slow
    def test_diagnostics_with_bgnbd_mcmc(self):
        """Test diagnostics work with real BG/NBD MCMC model."""
        from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper

        # Create small test dataset
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(20)],
                "frequency": np.random.randint(0, 10, size=20),
                "recency": np.random.uniform(0.0, 90.0, size=20),
                "T": np.full(20, 90.0),
            }
        )

        # Fit MCMC model (small chains for speed)
        config = BGNBDConfig(method="mcmc", chains=2, draws=200, tune=200)
        wrapper = BGNBDModelWrapper(config)
        wrapper.fit(data)

        # Test convergence check
        diagnostics = check_mcmc_convergence(wrapper.model.idata)
        assert isinstance(diagnostics, ConvergenceDiagnostics)
        assert isinstance(diagnostics.converged, bool)
        assert diagnostics.max_r_hat > 0.0

        # Test trace plot
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bg_nbd_trace.png"
            plot_trace_diagnostics(wrapper.model.idata, str(output_path))
            assert output_path.exists()

    @pytest.mark.slow
    def test_diagnostics_with_gamma_gamma_mcmc(self):
        """Test diagnostics work with real Gamma-Gamma MCMC model."""
        from customer_base_audit.models.gamma_gamma import (
            GammaGammaConfig,
            GammaGammaModelWrapper,
        )

        # Create small test dataset (requires frequency >= 2)
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(20)],
                "frequency": np.random.randint(2, 10, size=20),
                "monetary_value": np.random.uniform(20.0, 100.0, size=20),
            }
        )

        # Fit MCMC model
        config = GammaGammaConfig(method="mcmc", chains=2, draws=200, tune=200)
        wrapper = GammaGammaModelWrapper(config)
        wrapper.fit(data)

        # Test convergence check
        diagnostics = check_mcmc_convergence(wrapper.model.idata)
        assert isinstance(diagnostics, ConvergenceDiagnostics)

        # Test trace plot
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gamma_gamma_trace.png"
            plot_trace_diagnostics(wrapper.model.idata, str(output_path))
            assert output_path.exists()

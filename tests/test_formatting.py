from exohelp.formatting import (
    decimal_places_from_sigfigs,
    _format_scientific_latex,
    format_value_with_uncertainty,
    _use_adaptive_sigfigs,
)

# ---------------------------------------------------------------------------
# _format_scientific_latex
# ---------------------------------------------------------------------------


class TestLatexFloat:
    def test_small_number(self):
        assert _format_scientific_latex(1e-5) == r"10^{-5}"

    def test_small_with_coefficient(self):
        # _format_scientific_latex uses {:.1g} → 1 sig fig, so coefficient is always a single digit.
        # Exponent must be < -4 to trigger scientific notation; use 3e-5.
        assert _format_scientific_latex(3e-5) == r"3 \times 10^{-5}"

    def test_large_power_of_ten(self):
        assert _format_scientific_latex(1e3) == r"10^{3}"

    def test_non_scientific(self):
        # 0.5 is not in scientific notation
        assert "e" not in "{0:.1g}".format(0.5)
        assert _format_scientific_latex(0.5) == "0.5"


# ---------------------------------------------------------------------------
# decimal_places_from_sigfigs
# ---------------------------------------------------------------------------


class TestDeterminePrecision:
    def test_zero_returns_zero(self):
        assert decimal_places_from_sigfigs(0, significant_digits=1) == 0
        assert decimal_places_from_sigfigs(0, significant_digits=2) == 0

    def test_integer_one(self):
        # floor(log10(1)) = 0 → 1 - 1 - 0 = 0
        assert decimal_places_from_sigfigs(1.0, significant_digits=1) == 0

    def test_tenth(self):
        # floor(log10(0.1)) = -1 → 1 - 1 - (-1) = 1
        assert decimal_places_from_sigfigs(0.1, significant_digits=1) == 1

    def test_hundredth(self):
        # floor(log10(0.05)) = -2 → 1 - 1 - (-2) = 2
        assert decimal_places_from_sigfigs(0.05, significant_digits=1) == 2

    def test_thousandth(self):
        # floor(log10(0.0035)) = -3 → 1 - 1 - (-3) = 3
        assert decimal_places_from_sigfigs(0.0035, significant_digits=1) == 3

    def test_large_number_clamps_to_zero(self):
        # floor(log10(123)) = 2 → 1 - 1 - 2 = -2 → max(0, -2) = 0
        assert decimal_places_from_sigfigs(123.0, significant_digits=1) == 0

    def test_two_sig_digits_tenth(self):
        # 2 - 1 - (-1) = 2
        assert decimal_places_from_sigfigs(0.1, significant_digits=2) == 2

    def test_two_sig_digits(self):
        # floor(log10(0.15)) = -1 → 2 - 1 - (-1) = 2
        assert decimal_places_from_sigfigs(0.15, significant_digits=2) == 2

    def test_two_sig_digits_small(self):
        # floor(log10(0.024)) = -2 → 2 - 1 - (-2) = 3
        assert decimal_places_from_sigfigs(0.024, significant_digits=2) == 3

    def test_negative_number(self):
        # Uses abs(number); floor(log10(0.1)) = -1 → 1 - 1 - (-1) = 1
        assert decimal_places_from_sigfigs(-0.1, significant_digits=1) == 1


# ---------------------------------------------------------------------------
# adaptive_sigfigs  (new function)
# ---------------------------------------------------------------------------


class TestSmartSignificantDigits:
    """Rule: 2 sig figs when first digit of uncertainty is 1 or 2, else 1."""

    # First digit = 1 → 2 sig figs
    def test_first_digit_1_small(self):
        assert _use_adaptive_sigfigs(0.1) == 2

    def test_first_digit_1_with_fraction(self):
        assert _use_adaptive_sigfigs(0.12) == 2

    def test_first_digit_1_upper_bound(self):
        assert _use_adaptive_sigfigs(0.19) == 2

    def test_first_digit_1_large_scale(self):
        # 1.5 → first digit = 1
        assert _use_adaptive_sigfigs(1.5) == 2

    # First digit = 2 → 2 sig figs
    def test_first_digit_2(self):
        assert _use_adaptive_sigfigs(0.2) == 2

    def test_first_digit_2_with_fraction(self):
        assert _use_adaptive_sigfigs(0.024) == 2

    def test_first_digit_2_large_scale(self):
        assert _use_adaptive_sigfigs(2.4) == 2

    # First digit ≥ 3 → 1 sig fig
    def test_first_digit_3(self):
        assert _use_adaptive_sigfigs(0.3) == 1

    def test_first_digit_5(self):
        assert _use_adaptive_sigfigs(0.50) == 1

    def test_first_digit_9(self):
        assert _use_adaptive_sigfigs(0.9) == 1

    def test_first_digit_3_large_scale(self):
        assert _use_adaptive_sigfigs(3.7) == 1

    # Edge cases
    def test_zero_returns_one(self):
        assert _use_adaptive_sigfigs(0) == 1

    def test_negative_uncertainty(self):
        # Uses abs value; -0.15 → first digit = 1 → 2
        assert _use_adaptive_sigfigs(-0.15) == 2

    # Combined: adaptive_sigfigs + decimal_places_from_sigfigs reproduce the examples from the rule
    def test_pm_012(self):
        # ±0.12: first digit = 1 → 2 sig figs → decimal_places = 2
        sig = _use_adaptive_sigfigs(0.12)
        assert sig == 2
        assert decimal_places_from_sigfigs(0.12, significant_digits=sig) == 2

    def test_pm_024(self):
        # ±0.024: first digit = 2 → 2 sig figs → decimal_places = 3
        sig = _use_adaptive_sigfigs(0.024)
        assert sig == 2
        assert decimal_places_from_sigfigs(0.024, significant_digits=sig) == 3

    def test_pm_0004(self):
        # ±0.0004: first digit = 4 → 1 sig fig → decimal_places = 4
        sig = _use_adaptive_sigfigs(0.0004)
        assert sig == 1
        assert decimal_places_from_sigfigs(0.0004, significant_digits=sig) == 4


# ---------------------------------------------------------------------------
# format_value_with_uncertainty
# ---------------------------------------------------------------------------


class TestFormatValueWithUncertainty:
    def test_pm_basic(self):
        # std=0.06 → decimal_places=2 → 2 decimals
        assert format_value_with_uncertainty(0.57, 0.06, significant_digits=1) == r"$0.57 \pm 0.06$"

    def test_pm_larger_uncertainty(self):
        # std=0.45 → decimal_places=1 → 1 decimal; 0.45 rounds to 0.5
        assert format_value_with_uncertainty(1.23, 0.45, significant_digits=1) == r"$1.2 \pm 0.5$"

    def test_pm_half_uncertainty(self):
        assert format_value_with_uncertainty(12.34, 0.5, significant_digits=1) == r"$12.3 \pm 0.5$"

    def test_pm_scientific_notation_threshold(self):
        # decimal_places=7 > threshold=5 → _format_scientific_latex path
        assert (
            format_value_with_uncertainty(0.5, 1e-7, significant_digits=1)
            == r"$0.5000000 \pm 10^{-7}$"
        )

    def test_pm__use_adaptive_sigfigs(self):
        # std=0.12 → first digit=1 → 2 sig figs → decimal_places=2
        assert (
            format_value_with_uncertainty(0.57, 0.12, adaptive_sigfigs=True) == r"$0.57 \pm 0.12$"
        )

    def test_asymmetric_equal_collapses_to_pm(self):
        # equal diffs → pm_if_equal=True → ± format
        assert (
            format_value_with_uncertainty(
                0.57, 0.06, 0.06, significant_digits=1, uncertainty_style="asymmetric"
            )
            == r"$0.57 \pm 0.06$"
        )

    def test_asymmetric_different(self):
        # upper_diff=0.10 (prec=1), lower_diff=0.06 (prec=2) → mean uses coarser prec=1
        assert (
            format_value_with_uncertainty(
                0.57, 0.06, 0.10, significant_digits=1, uncertainty_style="asymmetric"
            )
            == r"$0.6^{+0.1}_{-0.06}$"
        )

    def test_asymmetric_pm_if_equal_false(self):
        # equal diffs but pm_if_equal=False → keep asymmetric notation
        assert (
            format_value_with_uncertainty(
                0.57,
                0.06,
                0.06,
                significant_digits=1,
                uncertainty_style="asymmetric",
                pm_if_equal=False,
            )
            == r"$0.57^{+0.06}_{-0.06}$"
        )

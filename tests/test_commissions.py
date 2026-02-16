"""Tests for the commissions module to ensure backwards compatibility and correct behavior."""

import aptrade as bt


class TestCommissionFactories:
    """Test factory functions for commission creation."""

    def test_futures_default(self):
        """Test basic futures() creates fixed commission futures."""
        comm = bt.commissions.futures()
        assert comm.p.stocklike is False
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_FIXED

    def test_futures_perc(self):
        """Test futures_perc() creates percentage commission futures."""
        comm = bt.commissions.futures_perc()
        assert comm.p.stocklike is False
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_PERC

    def test_futures_fixed(self):
        """Test futures_fixed() creates fixed commission futures."""
        comm = bt.commissions.futures_fixed()
        assert comm.p.stocklike is False
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_FIXED

    def test_stocks_default(self):
        """Test basic stocks() creates percentage commission stocks."""
        comm = bt.commissions.stocks()
        assert comm.p.stocklike is True
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_PERC

    def test_stocks_perc(self):
        """Test stocks_perc() creates percentage commission stocks."""
        comm = bt.commissions.stocks_perc()
        assert comm.p.stocklike is True
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_PERC

    def test_stocks_fixed(self):
        """Test stocks_fixed() creates fixed commission stocks."""
        comm = bt.commissions.stocks_fixed()
        assert comm.p.stocklike is True
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_FIXED

    def test_futures_with_custom_params(self):
        """Test futures() accepts additional parameters via kwargs."""
        comm = bt.commissions.futures(commtype=bt.commissions.CommInfoBase.COMM_PERC)
        assert comm.p.stocklike is False
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_PERC

    def test_stocks_with_custom_params(self):
        """Test stocks() accepts additional parameters via kwargs."""
        comm = bt.commissions.stocks(commtype=bt.commissions.CommInfoBase.COMM_FIXED)
        assert comm.p.stocklike is True
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_FIXED


class TestLegacyCommissionClasses:
    """Test legacy class-based commission API for backwards compatibility."""

    def test_comminfo_futures(self):
        """Test CommInfo_Futures class."""
        comm = bt.commissions.CommInfo_Futures()
        assert comm.p.stocklike is False

    def test_comminfo_futures_perc(self):
        """Test CommInfo_Futures_Perc class."""
        comm = bt.commissions.CommInfo_Futures_Perc()
        assert comm.p.stocklike is False
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_PERC

    def test_comminfo_futures_fixed(self):
        """Test CommInfo_Futures_Fixed class."""
        comm = bt.commissions.CommInfo_Futures_Fixed()
        assert comm.p.stocklike is False
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_FIXED

    def test_comminfo_stocks(self):
        """Test CommInfo_Stocks class."""
        comm = bt.commissions.CommInfo_Stocks()
        assert comm.p.stocklike is True

    def test_comminfo_stocks_perc(self):
        """Test CommInfo_Stocks_Perc class."""
        comm = bt.commissions.CommInfo_Stocks_Perc()
        assert comm.p.stocklike is True
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_PERC

    def test_comminfo_stocks_fixed(self):
        """Test CommInfo_Stocks_Fixed class."""
        comm = bt.commissions.CommInfo_Stocks_Fixed()
        assert comm.p.stocklike is True
        assert comm.p.commtype == bt.commissions.CommInfoBase.COMM_FIXED


class TestCommissionEquivalence:
    """Test that factory functions produce equivalent results to legacy classes."""

    def test_futures_perc_equivalence(self):
        """Test futures_perc() == CommInfo_Futures_Perc."""
        factory = bt.commissions.futures_perc()
        legacy = bt.commissions.CommInfo_Futures_Perc()

        assert factory.p.stocklike == legacy.p.stocklike
        assert factory.p.commtype == legacy.p.commtype

    def test_futures_fixed_equivalence(self):
        """Test futures_fixed() == CommInfo_Futures_Fixed."""
        factory = bt.commissions.futures_fixed()
        legacy = bt.commissions.CommInfo_Futures_Fixed()

        assert factory.p.stocklike == legacy.p.stocklike
        assert factory.p.commtype == legacy.p.commtype

    def test_stocks_perc_equivalence(self):
        """Test stocks_perc() == CommInfo_Stocks_Perc."""
        factory = bt.commissions.stocks_perc()
        legacy = bt.commissions.CommInfo_Stocks_Perc()

        assert factory.p.stocklike == legacy.p.stocklike
        assert factory.p.commtype == legacy.p.commtype

    def test_stocks_fixed_equivalence(self):
        """Test stocks_fixed() == CommInfo_Stocks_Fixed."""
        factory = bt.commissions.stocks_fixed()
        legacy = bt.commissions.CommInfo_Stocks_Fixed()

        assert factory.p.stocklike == legacy.p.stocklike
        assert factory.p.commtype == legacy.p.commtype

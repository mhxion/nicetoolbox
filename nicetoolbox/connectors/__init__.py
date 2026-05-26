"""NICE Connectors: import/export bridges to third-party labelling and tracking tools.

Each connector parses or writes files produced/consumed by an external tool and
converts them to/from NICE Toolbox NPZ components. The resulting NPZ files share
the same structure and data_description convention as detector outputs, so the
visualizer and evaluation modules work on connector outputs unchanged.
"""

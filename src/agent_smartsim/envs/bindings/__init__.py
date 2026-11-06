"""Agent-tool bindings for third-party simulators.

Each module here adapts one simulator to the five-method EnvBinding
contract (see ../binding.py). Nothing in this package imports a simulator
at module import time -- bindings import lazily so the rest of the repo
stays usable (and testable) on machines without the sim installed.
"""

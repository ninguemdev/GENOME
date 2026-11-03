"""Pure, deterministic simulation core (ARQUITETURA §2, §3 to §7).

Holds configuration, random streams, toroidal geometry, world state and the
tick systems. It imports no other GENOME package: it declares protocols
(``genome.core.protocols``) that other packages implement and that
``genome.lab`` injects.
"""

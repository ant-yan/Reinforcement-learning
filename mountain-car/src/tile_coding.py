from math import floor
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Union


class IHT:
    """
    Index Hash Table (IHT).

    This structure maps arbitrary coordinate tuples to integer indices in
    the range [0, size). It is typically used in tile coding to turn a
    set of active tiles into feature indices. When the capacity is
    exceeded, new coordinates are mapped by hashing, which may introduce
    collisions.
    """

    def __init__(self, size_val: int) -> None:
        """
        Initialize the IHT.

        Parameters
        ----------
        size_val : int
            Maximum number of distinct entries before collisions are forced.
        """
        self.size: int = size_val
        self.overfull_count: int = 0
        self.dictionary: Dict[Hashable, int] = {}

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """
        Return the number of distinct coordinates currently stored.
        """
        return len(self.dictionary)

    def full(self) -> bool:
        """
        Check whether the IHT has reached its capacity.
        """
        return self.count() >= self.size

    def get_index(self, obj: Hashable, read_only: bool = False) -> Optional[int]:
        """
        Return the index associated with `obj`.

        If `obj` is already present, its existing index is returned.
        If it is not present and `read_only` is True, `None` is returned.
        If it is not present and the table is not yet full, a new index is
        created and returned.
        If the table is full, the index is computed via hashing
        (allowing collisions).

        Parameters
        ----------
        obj : Hashable
            Coordinate key (usually a tuple) describing a tile.
        read_only : bool, optional
            If True, do not insert new entries; return None instead.

        Returns
        -------
        int or None
            The index for `obj`, or None if `read_only` is True and `obj`
            is not stored.
        """
        d = self.dictionary

        # Existing entry
        if obj in d:
            return d[obj]

        # Read-only lookup
        if read_only:
            return None

        # Table still has capacity
        if not self.full():
            new_index = self.count()
            d[obj] = new_index
            return new_index

        # Table is full: fall back to hashed index (collision possible)
        if self.overfull_count == 0:
            print("IHT full, starting to allow collisions")
        self.overfull_count += 1
        return hash(obj) % self.size


# ---------------------------------------------------------------------- #
# Standalone functions
# ---------------------------------------------------------------------- #

def hash_coords(
    coordinates: Sequence[int],
    m: Union[IHT, int, None],
    read_only: bool = False,
) -> Union[int, Sequence[int], None]:
    """
    Map a coordinate sequence to an integer index.

    Parameters
    ----------
    coordinates : sequence of int
        Coordinate values describing a particular tile.
    m : IHT or int or None
        - If an `IHT` instance: use it to obtain or assign an index.
        - If an `int`: treat it as the modulus and return `hash(coords) % m`.
        - If `None`: return the coordinates as-is.
    read_only : bool, optional
        If True and `m` is an IHT, do not create new entries.

    Returns
    -------
    int or sequence of int or None
        - Integer index when using an IHT or integer modulus.
        - The original coordinates when `m` is None.
    """
    coords_tuple = tuple(coordinates)

    if isinstance(m, IHT):
        return m.get_index(coords_tuple, read_only=read_only)

    if isinstance(m, int):
        return hash(coords_tuple) % m

    # When m is None, just propagate the coordinates (useful for debugging)
    return coordinates


def tiles(
    iht_or_size: Union[IHT, int, None],
    num_tilings: int,
    floats: Sequence[float],
    ints: Optional[Iterable[int]] = None,
    read_only: bool = False,
) -> List[Optional[int]]:
    """
    Map continuous and discrete variables to a list of active tile indices.

    This function implements tile coding: it overlays multiple offset grids
    (tilings) over the space of float variables and returns, for each tiling,
    the index of the corresponding active tile.

    Parameters
    ----------
    iht_or_size : IHT or int or None
        - IHT: use this index hash table to obtain tile indices.
        - int: size of the index space; indices are `hash(coords) % size`.
        - None: coordinates themselves are returned instead of indices.
    num_tilings : int
        Number of tilings to use. Typically a power of two and at least
        4 times the number of float variables.
    floats : sequence of float
        Continuous variables. They are scaled by `num_tilings` so that each
        tiling defines a grid with approximately unit generalization.
    ints : iterable of int, optional
        Additional discrete variables to include in the tile coordinates.
    read_only : bool, optional
        If True and `iht_or_size` is an IHT, do not create new entries.

    Returns
    -------
    list of int or None
        A list of length `num_tilings`, containing the active tile index
        for each tiling. If `iht_or_size` is None, the raw coordinate
        tuples (or their hashed forms) may be returned instead.
    """
    if ints is None:
        ints = []

    # Quantize floats relative to the number of tilings
    quantized_floats = [floor(f * num_tilings) for f in floats]

    active_tiles: List[Optional[int]] = []

    for tiling in range(num_tilings):
        # Each tiling gets its own offset
        tiling_offset = tiling * 2

        # Start coordinates with tiling index (to separate grids)
        coords = [tiling]
        bias = tiling

        # Offset and downscale the quantized float values
        for q in quantized_floats:
            coords.append((q + bias) // num_tilings)
            bias += tiling_offset

        # Append any provided integer features
        coords.extend(ints)

        # Map coordinates to an index
        active_tiles.append(hash_coords(coords, iht_or_size, read_only=read_only))

    return active_tiles

from ufl.corealg.dag_traverser import DAGTraverser
from ufl.classes import (
    Expr,
    ReferenceGrad,
    SpatialCoordinate
    )
from functools import singledispatchmethod
from ufl import dot, grad

class MeshTransformPullbacks(DAGTraverser):
    """
    DAGTraversor to pull back equations from a transformed mesh,
    defined by the transformation x -> Phi(x).
    Phi is provided as a Firedrake Function or UFL expression.
    """

    def __init__(
        self,
        Phi: Expr,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr] | None = None,
        result_cache: dict[Expr, Expr] | None = None,
    ) -> None:
        """Initialise.

        Args:
            Phi: expression describing map to deformed mesh, x -> Phi(x).
            compress: If True, ``result_cache`` will be used.
            visited_cache: cache of intermediate results; expr -> r = self.process(expr, ...).
            result_cache: cache of result objects for memory reuse, r -> r.
        """
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        self._Phi = Phi

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """
        Pull back the expression using Phi.
        Args:
            o: UFL Expression to be processed.
        """
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr ) -> Expr:
        """Handle Expr."""
        return self.reuse_if_untouched(o)

    @process.register(ReferenceGrad)
    def _(self, o: Expr ) -> Expr:
        """Handle ReferenceGrad.
        If it is applied to SpatialCoordinate,
        multiply by Grad(Phi), otherwise don't.
        """
        print(o)
        if isinstance(o, SpatialCoordinate):
            print("here")
            return dot(grad(self.Phi).T, o)
        else:
            print("there")
            return self.reuse_if_untouched(o)


# Frozen administrative graph construction

This methods record follows Section 2.6 of the submitted manuscript. It documents the reported graph definition; source polygons, frozen edge arrays, and the executable graph-builder are not included in this model-only repository.

## Nodes and GEO1

The node set is the fixed set of 3,236 ADM1-equivalent administrative units. GEO1 uses strict shared-boundary adjacency: two polygons receive an edge only if their boundaries intersect and the shared-boundary length, measured on the WGS84 ellipsoid, is at least 0.01 km. Point-only contact and shared-boundary contact shorter than that threshold do not create a GEO1 edge.

A prespecified cross-border topology repair accommodates small geometric offsets in global administrative boundaries. An additional cross-border adjacency is permitted only when the two polygons have limited areal overlap, the overlap is no more than 5% of the smaller unit's area, and neither unit's representative point lies inside the other polygon.

The frozen GEO1 graph contains 7,427 undirected edges, represented as 14,854 directed relations. It includes 1,553 undirected cross-border edges spanning 311 country pairs; 182 nodes have no GEO1 neighbors. No nearest-neighbor edges are added for isolated nodes.

## GEO2 and GEO3

GEO2 and GEO3 are disjoint exact-hop neighborhoods derived from shortest-path distance on the frozen GEO1 topology. GEO2 contains pairs at shortest-path distance exactly two, excluding GEO1 pairs. GEO3 contains pairs at distance exactly three, excluding GEO1 and GEO2 pairs. They contain 30,970 and 48,054 directed relations, respectively.

## Interpretation and provenance

The graph is administrative geographic context. Its hop count is not a uniform physical distance, and the edges do not represent animal movement, trade, or pathogen transmission. Construction was frozen before predictive evaluation and did not use disease labels, predictions, or validation performance.

These details are transcribed from the submitted Methods section. The underlying polygon inputs, edge list, executable builder, geometry-repair implementation, and hash validation are not released here; therefore this page documents the manuscript's graph specification but does not make graph construction computationally reproducible from this repository alone.

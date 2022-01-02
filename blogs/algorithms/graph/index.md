---
title: Graph
description: An example of a subfolder page.
level: two
cat: algorithms
---

# Graph

<div class="section-index">
    <hr class="panel-line">
    {% for file in site.pages  %}
        {% if file.level == "three" and file.cat == "graph" %}
            <div class="entry">
                <h5><a href="{{ file.url | prepend: site.baseurl }}">{{ file.title }}</a></h5>
                <p>{{ file.description }}</p>
            </div>
        {% endif %}
    {% endfor %}
</div>



**Introduction, DFS and BFS**

> Graph and its representations
> Breadth First Traversal for a Graph
> Depth First Traversal for a Graph
> Applications of Depth First Search
> Applications of Breadth First Traversal
> Graph representations using set and hash
> Find Mother Vertex in a Graph
> Transitive Closure of a Graph using DFS
> Find K cores of an undirected Graph
> Iterative Depth First Search
> Count the number of nodes at given level in a tree using BFS
> Count all possible paths between two vertices
> Minimum initial vertices to traverse whole matrix with given conditions
> Shortest path to reach one prime to other by changing single digit at a time
> Water Jug problem using BFS
> Count number of trees in a forest
> BFS using vectors & queue as per the algorithm of CLRS
> Level of Each node in a Tree from source node
> Construct binary palindrome by repeated appending and trimming
> Transpose graph
> Path in a Rectangle with Circles
> Height of a generic tree from parent array
> BFS using STL for competitive coding
> DFS for a n-ary tree (acyclic graph) represented as adjacency list
> Maximum number of edges to be added to a tree so that it stays a Bipartite graph
> A Peterson Graph Problem
> Implementation of Graph in JavaScript
> Print all paths from a given source to a destination using BFS
> Minimum number of edges between two vertices of a Graph
> Count nodes within K-distance from all nodes in a set
> Bidirectional Search
> Minimum edge reversals to make a root
> BFS for Disconnected Graph
> Move weighting scale alternate under given constraints
> Best First Search (Informed Search)
> Number of pair of positions in matrix which are not accessible
> Maximum product of two non-intersecting paths in a tree
> Delete Edge to minimize subtree sum difference
> Find the minimum number of moves needed to move from one cell of matrix to another
> Minimum steps to reach target by a Knight | Set 1
> Minimum number of operation required to convert number x into y
> Minimum steps to reach end of array under constraints
> Find the smallest binary digit multiple of given number
> Roots of a tree which give minimum height
> Stepping Numbers
> Clone an Undirected Graph
> Sum of the minimum elements in all connected components of an undirected graph
> Check if two nodes are on same path in a tree
> A matrix probability question
> Find length of the largest region in Boolean Matrix
> Iterative Deepening Search(IDS) or Iterative Deepening Depth First Search(IDDFS)



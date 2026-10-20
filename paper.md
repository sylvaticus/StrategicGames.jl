
# UNDER IMPLEMENTATION !!!

---
title: 'StrategicGames: a julia package to work with strategic games'
tags:
  - Julia
  - Game Theory
  - Nash Equilibrium
authors:
  - name: Antonello Lobianco^[Corresponding author.]
    orcid: 0000-0002-1534-8697
    affiliation: "1, 2, 3, 4, 5, 6" # (Multiple affiliations must be quoted)
affiliations:
 - name: Université de Lorraine
   index: 1
 - name: Université de Strasbourg
   index: 2
 - name: Institut des sciences et industries du vivant et de l'environnement (AgroParisTech)
   index: 3
 - name: Centre national de la recherche scientifique (CNRS)
   index: 4
 - name: Institut national de recherche pour l’agriculture, l’alimentation et l’environnement (INRAE)
   index: 5
 - name: Bureau d'économie théorique et appliquée (BETA)
   index: 6
date: XX April 2023
bibliography: docs/paper/paper.bib
---

<!-- Test it with:

`pandoc --filter pandoc-citeproc --bibliography docs/paper/paper.bib  paper.md -o paper.pdf`

-->

# Summary

A series of functions to work with strategic games (currently only in normal form) has been implemented and bundled together with several "utility" functions in a single package for the Julia programming language.
Currently functions are available to retrieve the payoff, the best strategies (conditional to the other players strategies) and the Nash equilibrium(a) for generic N-players game

This paper presents the motivations and the general approach of the package and gives an overview of its organisation. We refer the reader to the [package documentation](https://sylvaticus.github.io/StrategicGames.jl/stable) for instructions on how to use the various algorithms provided or to the notes available on GitHub [@Lobianco:2022] for a more wide discussion.


# Motivations and objectives



# Acknowledgements

This work was supported by a grant overseen by the French National Research Agency (ANR) as part of the "Investissements d'Avenir" program (ANR-11-LABX-0002-01, Lab of Excellence ARBRE).

# References

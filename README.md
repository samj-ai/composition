# composition

Develop and test compositional generalization abilities in nn's  

This libary consists of sandboxes / benchmark tasks to evaluate compositional generalization in nn's.

 - ARC-AGI
 - ARC-AGI-2
 - Szpakowski patterns
 - SRaven's matrices
 - BBH

Moreover, here I also experiment with dimensions along with transformers and training can be modified to facilitate compositional generalization (and more generally, data efficiency).

Some context in a writeup [here](https://samj-ai.github.io/2025/03/26/ARC-AGI.html).

Usage notes:

- For handling many external data sources, I use git submodules, so if you want to clone this repo, make sure you do:

```
git clone --recurse-submodules git@github.com:samj-ai/composition.git
```

- The same interface in data/ARC-AGI/apps can be used to visualize SRaven tasks as well as the "traces" of Szpakowski tasks.
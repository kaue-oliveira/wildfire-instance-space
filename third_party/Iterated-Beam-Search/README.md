# Iterated-Beam-Search-for-Wildfire-Suppression

Accompanying repository of the paper "Iterated Beam Search for Wildfire Suppression". 


### Building & Compiling & Running
The codebase can be compiled by running the script  ```configure.sh```. You will need to have [Boost](https://www.boost.org/) installed.

```bash
./configure.sh
```

You can run the algorithm on the instance ```LA0``` by typing
```bash
./build/fire --instance ./instances/L0_a.json --beta 50 --eta 70 --c 30 --p 0.5 --zmax 3 --timelimit 60 --seed 1
```

### Decimal WSP times (local TCC extension)

The instance reader also accepts the decimal Harris inputs in the TCC's 19
supplementary Benchmark74 instances. JSON numbers and resource-time keys must
use a period, for example `123.13`; a comma is not a JSON decimal separator.
The reader detects fractional horizon, delay, release times or arc weights and
stores **all** temporal quantities as exact integer hundredths. Thus `123.13`
becomes `12313` internally and is printed again as `123.13` in STN signatures
and saved solutions. Values not representable in hundredths, ambiguous
resource instants after scaling, or arithmetic bounds that might overflow are
rejected instead of being silently truncated. The historical integer instances
continue to use scale 1.

This source change is for future runs. It does not alter the archived
Benchmark74 v2.4 trajectories, which were produced by the old reader and are
kept intact. The TCC final report documents a separate, read-only revaluation
of every state archived in those trajectories under the decimal instance.

### Running the experiments
Type the following commands to reproduce our results:
```bash
cd experiments
./exec.sh
```
The codebase will be compiled and our algorithm will run on all 16 instances in ```./instances```. You will need to have [GNU Parallel](https://www.gnu.org/software/parallel/) installed.


### Instances

In the directory ```./instances``` you will find the 16 instances used in the paper. These instances are originally from
[this repository](https://github.com/mitchopt/Logic-based-Benders-decomposition-for-wildfire-suppression). 


## How to cite

TBD.

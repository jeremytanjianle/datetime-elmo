# Natural Date String Parser
A sequence-to-sequence model to parse natural date strings into machine parsable date strings.

| Input                | Output                    |
|----------------------|---------------------------|
| 1st of March 2019    | 01/03/2019 eos            |
| 15 Febroory 2008     | 15/02/2008 eos            |
| 19 Sept - 27 Sept 99 | 19/08/1999 - 27/08/99 eos |
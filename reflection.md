Runtime analysis:
Block Size was 32 for all. I ran the test 3 times each and took the average

| Size   | CPU        | Naive GPU   | MM Tile       | Culbas     |
|--------|------------|-------------|---------------|------------|
| 128    | 0.007537 ns| 0.000414 ns | 0.370979 ms   | 0.000831 ns|
| 256    | 0.098184 ns| 0.000538 ns | 0.521183 ms   | 0.000890 ns|
| 1024   | 5.872243 ns| 0.006722 ns | 3.972054 ms   | 0.002922 ns|

0. My results sort of make sense, except for my Tile, I think something is wrong with that one but I am not sure what. But otherwise it makes sense that as the size increases the CPU is slowest then Naive GPU, then Cublas. 

1. Cublas got faster as the size input got larger, so my speed improved.
2. I think implementing cublas went well, adn the naive CPU
3. I struggled with the Tile, i understand the concept and thoguht I had implemented it corrctly, but my time is so slow it is very confusing, 
4. I would look deeper into my tile and see if i can figure out what is going on with the timings
5. na <3 :)
# KWEST Matching Program
Code to match Kellogg students to a KWEST trip based on submitted preferences.

#### ToDo
1. Create 'none' weight for top trips
1. Fix the 'any' preference prediction

#### Steps
1. Execute `run.sh` shell file

#### Methodology
1. Import data
1. Eliminate unpopular trips based on student input
1. Predict missing trip preferences using `K-Nearest Neighbors Regression Algorithm`
1. Generate potential trip matches using `Hospital-Resident Matching Algorithm`
1. Pick match with trips that most closely represent the demographs of the overall student population
1. Write output to CSV file

#### Suggestions
- Require all students to rank 10 or more trips to increase likelihood of optimal matches

#### Documentation
- `Kwest`
    - Central object for orchestration
    - Methods
        - `setup`
            - `trip_capacity` parameter - sets the max number of students that can be assigned in a trip
        - `predict`
            - `weight` parameter - 
                - 'none' - each trip vote is assigned 1 point
                - 'linear' - trips are weighted based on vote rank, where rank 1 received 10 points and rank 10 receives 1 point
                - 'exponential' - trip votes are weighted based on vote rank, where rank 1 receives exp(1) and rank 10 received exp(0.1)
            - `preference` parameter - 
                - 'stated' - stated preferences are preferred to predicted
                - 'any' - stated and predicted preferences are treated equally
        - `match`
            - `runs` parameter - the number of potential trip assignment solutions to generate
        - `pick` - chooses the best match iteration based on match preference
            - `preference` parameter
                - 'match' - chooses match with least "mismatches" or cases where an assigned trip was not in stated preferences
                - 'demographics' - chooses match with trips that have demographics that align most closely with population demographics
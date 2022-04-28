# KWEST Matching Program
Code to match Kellogg students to a KWEST trip based on submitted preferences.

#### ToDo
1. N/A

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
    - Parameters
        - `trip_capacity` - sets the max number of students that can be assigned in a trip
    - Methods
        - `setup` - cleans imported data, creates students and trips
            - Parameters
                - `weight` - designates how the list of top trips is decided
                    - Options
                        - `none` - each trip vote is assigned 1 point
                        - `linear` - trips are weighted based on vote rank, where rank 1 received 10 points and rank 10 receives 1 point
                        - `exponential` - trip votes are weighted based on vote rank, where rank 1 receives exp(1) and rank 10 received exp(0.1)
        - `predict` - predicts preferences for trips that a student didn't vote on
            - Parameters
                - `weight` - designates how the trip score field is calculated
                    - Options
                        - `linear` - trip score is based on vote rank, where rank 1 received 10 points and rank 10 receives 1 point
                        - `exponential` - trip score is based on exponentiated vote rank, where rank 1 receives exp(1) and rank 10 received exp(0.1)
                - `preference` - designates how trips are ranked
                    - Options
                        - `stated` - stated preferences are preferred to predicted
                        - `any` - stated and predicted preferences are treated equally
        - `match` - generates a solution to the trip assignment problem
            - Parameters
                - `runs` - the number of potential trip assignment solutions to generate
        - `pick` - chooses the best match iteration based on match preference
            - Parameters
                - `preference` - designates on what metric a best match is selected
                    - Options
                        - `match` - chooses match with least "mismatches" or cases where an assigned trip was not in stated preferences
                        - `demographics` - chooses match with trips that have demographics that align most closely with population demographics
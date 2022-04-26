# KWEST Matching Program
Code to match Kellogg students to a KWEST trip based on submitted preferences.

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

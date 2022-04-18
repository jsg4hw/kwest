# KWEST Matching Program
Code to match Kellogg students to a KWEST trip based on submitted preferences.

#### Steps
1. Change `INPUT_FPATH` and `OUTPUT_FPATH` to location of input trip preferences excel file and preferred output location for CSV file of trip rosters
1. Run all code cells

#### Methodology
1. Import data
1. Eliminate unpopular trips based on student input
1. Predict missing trip preferences using **K-Nearest Neighbors Regression Algorithm**
1. Generate potential trip matches using **Hospital-Resident Matching Algorithm**
1. Pick match with trips that most closely represent the demographs of the overall student population
1. Write output

#### Suggestions
- Require all students to submit 10 trip preferences - this increased likelihood of optimal trip matches

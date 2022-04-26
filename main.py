#!/usr/bin/env python
# coding: utf-8

# # ToDo
# 1. Replace `vote` column with `score` and calculate score based on function parameter - options: linear, exponential

# In[84]:


import datetime
import os
import operator
from random import shuffle
from math import ceil
from math import exp
from itertools import product
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from matching.games import HospitalResident


# In[103]:


class Kwest(object):
    def __init__(self,trip_capacity=20):
        
        cwd = os.getcwd()
        
        self.input_fpath = os.path.join(cwd,'data.xlsx')
        self.output_fpath = os.path.join(cwd,'output.csv')
        
        self.df_input = pd.read_excel(os.path.join(cwd,'data.xlsx'),dtype=str)
        self.df_clean = None
        self.df_final = None
        
        self.df_fit = None
        self.fit_x = None
        self.fit_y = None
        
        self.df_pred = None
        self.pred_x = None
        self.pred_y = None
        
        self.students = {}
        self.trips = []
        self.top_trips = []
        self.trip_capacity = trip_capacity
        
        self.gender_avg = None
        self.program_avg = None
        self.country_avg = None
        
        self.matches = None
        self.final_match = None
        
        self._clean_data()
    
    def _clean_data(self):
        df = self.df_input[[
            'Netid','Gender','Program','Date of Birth','Passport Country',
            'Sig Other Coming','Sig Other Kellogg Student',
        ]+['Vote'+str(i+1) for i in range(10)]]
        
        df.columns = ['net_id','gender','program','dob','country','sig other coming','sig other kellogg student']+['trip'+str(i+1) for i in range(10)]
        df.loc[:,'jv_coming'] = df.apply(lambda x: 1 if ((x['sig other coming']=='T') & (x['sig other kellogg student']=='F')) else 0, axis=1)
        self.df_clean = df.set_index('net_id').drop(['sig other coming','sig other kellogg student'],axis=1)
        
    def _wrangle_data(self):
        df = self.df_clean.drop(['dob'],axis=1) # DOB is null for all students in test data set
        df.gender = df.gender.apply(lambda x: 1 if x=='F' else 0)
        df.program = df.program.apply(lambda x: 1 if x=='2YMBA' else 0)
        df.country = df.country.apply(lambda x: 1 if x=='UNITED STATES' else 0)
        
        fit = pd.melt(
            df.reset_index(drop=False),
            id_vars=['net_id','gender','country','program','jv_coming'],
            value_vars=['trip'+str(i+1) for i in range(10)],
            var_name='vote',value_name='trip').dropna()
        fit.vote = fit.vote.str.replace('trip','').astype(int)
                
        netid_trips = list(product(df.index.tolist(), self.trips))
        # 1. df of all net id and trip name combinations
        # 2. join to find which trips have already been voted on
        # 3. join to pull student attribute data
        pred = pd.DataFrame(netid_trips,columns=['net_id','trip'])            .merge(fit[['net_id','trip','vote']],on=['net_id','trip'],how='left')            .merge(df.reset_index(drop=False)[['net_id','gender','country','program','jv_coming']],on=['net_id'],how='inner') 
        
        fit = fit.join(pd.get_dummies(fit.trip)).drop(labels=['trip'],axis=1).reset_index(drop=True)
        fit.index.name = 'row_num'
        fit = fit.reset_index(drop=False).set_index(['row_num','net_id'])
        cols_fit = fit.columns.tolist()
        self.df_fit = fit
        self.fit_x = fit.drop(['vote'],axis=1)
        self.fit_y = fit.vote
        
        pred = pred.loc[(pred.vote.isnull()) & (pred.trip.isin(self.top_trips))]
        pred = pred.join(pd.get_dummies(pred.trip)).drop(labels=['trip','vote'],axis=1).reset_index(drop=True)
        pred.index.name = 'row_num'
        pred = pred.reset_index(drop=False).set_index(['row_num','net_id'])
        cols_pred = [col for col in cols_fit if col!='vote']
        for col in cols_pred:
            if col not in pred.columns:
                pred[col] = 0
        pred = pred[cols_pred]
        self.df_pred = pred
        self.pred_x = pred
    
    def _generate_students(self):
        students = {}
        rows = self.df_clean.reset_index(drop=False).to_dict('records')
        trips = ['trip'+str(i+1) for i in range(10)]
        
        for student in rows:
            votes = []
            for trip in trips:
                vote = student.pop(trip,None)
                if str(vote) != 'nan':
                    votes.append(vote)
            student['votes'] = votes
            students[student['net_id']] = Student(**student)
        self.students = students
        
    def _demographics(self):
        total = len(self.students)
        gender = 0
        program = 0
        country = 0
        for student in self.students.values():
            if student.gender=='F':
                gender+=1
            if student.program=='2YMBA':
                program+=1
            if student.country=='UNITED STATES':
                country+=1
        self.gender_avg = (gender/total)
        self.program_avg = (program/total)
        self.country_avg = (country/total)
    
    def _generate_trips(self):
        cols = ['trip'+str(i+1) for i in range(10)]
        df = self.df_clean[cols]
        
        trips = [df[i].tolist() for i in cols]
        trips = set([t for trip in trips for t in trip if str(t) != 'nan'])
        self.trips = trips
    
    def _top_trips(self,weight):
        cushion = 1
        trip_goers = 0
        for student in self.students.values():
            if student.jv_coming == 1:
                trip_goers+=2
            else:
                trip_goers+=1
        min_capacity,max_capacity = 14,20
        min_trips,max_trips = (ceil(trip_goers/max_capacity)),(ceil(trip_goers/min_capacity))
        
        if weight in ['linear','exponential']:
            votes = {}
            for net_id,student in self.students.items():
                for i,trip in enumerate(student.votes):
                    if weight == 'linear':
                        try:
                            votes[trip] += (11-i)
                        except:
                            votes[trip] = (11-i)
                    elif weight == 'exponential':
                        try:
                            votes[trip] += exp(11-i)
                        except:
                            votes[trip] = exp(11-i)
            top_trips = sorted(votes.items(),key=operator.itemgetter(1),reverse=True)
            top_trips = [i[0] for i in top_trips][:min_trips+cushion]
        else:
            votes = []
            for net_id,student in self.students.items():
                votes+=student.votes
            top_trips = pd.Series(votes).value_counts()[:min_trips+cushion].index.tolist()
        self.top_trips = top_trips
    
    def setup(self,weight='linear'):
        
        self._generate_students()
        self._demographics()
        
        self._generate_trips()
        self._top_trips(weight)
    
    def predict(self,weight='exponential',preference='stated'):
        
        self._wrangle_data()
        
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(self.fit_x, self.fit_y)
        pred_y = knn.predict(self.pred_x)
        pred_y = pd.DataFrame(pred_y,columns=['vote'])
        pred_y.index.name = 'row_num'
        self.pred_y = pred_y
        
        fit = self.df_fit
        pred = self.pred_x.join(pred_y,on=['row_num'],how='inner')
        final = pd.concat([fit,pred])
        
        final = pd.melt(
            final.reset_index(drop=False).drop(['row_num'],axis=1),
            id_vars=['net_id','gender','country','program','jv_coming','vote'],
            var_name='trip'
        )
        final = final.loc[(final.value==1) & (final.trip.isin(self.top_trips))].drop(['value'],axis=1)
        self.df_final = final
        
        if preference == 'stated':
            None
        else:
            None
            
    def match(self,runs=10):
        matches = []
        student_preferences = {}
        preferences = self.df_final.sort_values(['net_id','vote']).groupby(['net_id']).agg(list)[['trip']]
        for net_id,student in self.students.items():
            prefs = preferences.loc[preferences.index==net_id].trip[0]
            student.preferences = prefs
            student_preferences[net_id] = prefs
            if student.jv_coming == 1:
                student_preferences[net_id+'JV'] = prefs        
        
        net_ids_w_jvs = list(student_preferences.keys())
        
        for run in range(runs):
            trip_preferences = {}
            for trip in self.top_trips:
                shuffle(net_ids_w_jvs)
                for net_id in net_ids_w_jvs:
                    if net_id[-2:] == 'JV':
                        net_ids_w_jvs.insert(
                            # in the spot directly after student partner
                            net_ids_w_jvs.index(net_id[:-2])+1,
                            # insert the jv netid
                            net_ids_w_jvs.pop(net_ids_w_jvs.index(net_id))
                        )
                trip_preferences[trip] = net_ids_w_jvs
                
            match = {
                'iteration': str(run+1),
                'student_mapper': self.students,
                'student_preferences': student_preferences,
                'trip_preferences': trip_preferences,
                'trip_capacity': self.trip_capacity,
                'gender_avg': self.gender_avg,
                'program_avg': self.program_avg,
                'country_avg': self.country_avg,
            }
            matches.append(Match(**match))
        self.matches = matches
    
    def pick(self,preference='match'):
        best = self.matches[0]
        
        if preference == 'match':
            None
        else:
            None
        
        for match in self.matches:
            if match.error < best.error:
                best = match
        self.final_match = best
        
        print("""
        Best Match Summary
        -------------------
        Iteration: {}
        Corrections: {}
        Error: {}
        """.format(
            self.final_match.iteration,
            self.final_match.corrections,
            round(self.final_match.error,2),
        ))
        # for name,trip in self.final_match.trips.items():
        #     print(name)
        #     print(int(100*trip.error))
        #     print(trip.size,len(trip.students),sum([1 for student in trip.students if student.jv_coming==1]))
        #     print([student.net_id for student in trip.students])
        #     print('')
        
        output = []
        for name,trip in self.final_match.trips.items():
            output.append([name]+[student.net_id for student in trip.students])
        pd.DataFrame(output).to_csv(self.output_fpath,header=False,index=False)


# In[104]:


class Match(object):
    def __init__(self,iteration,student_mapper,student_preferences,
                 trip_preferences,trip_capacity,
                 gender_avg,program_avg,country_avg,
                 solution=None,trips=None,
                 corrections=None,error=None):
        self.iteration = iteration
        self.student_mapper = student_mapper
        self.student_preferences = student_preferences
        
        self.trip_preferences = trip_preferences
        self.trip_capacity = trip_capacity
        
        self.gender_avg = gender_avg
        self.program_avg = program_avg
        self.country_avg = country_avg
        
        self.solution = solution
        self.trips = trips
        
        self.corrections = corrections
        self.error = error
        
        self._solve()
        self._correct()
        self.score()
    
    def _solve(self):
        trip_capacity = {trip:self.trip_capacity for trip in self.trip_preferences.keys()}
        game = HospitalResident.create_from_dictionaries(
            self.student_preferences,
            self.trip_preferences,
            trip_capacity
        )
        solution = game.solve(optimal='resident')
        self.solution = solution
        trips = {}
        
        for trip,students in solution.items():
            _trip = {
                'name': trip.name,
                'students': [],
                'gender_avg': self.gender_avg,
                'program_avg': self.program_avg,
                'country_avg': self.country_avg,
                'capacity': self.trip_capacity,
            }
            for student in students:
                if student.name[-2:]!='JV':
                    _trip['students'].append(self.student_mapper[student.name])
            trips[trip.name] = (Trip(**_trip))
        self.trips = trips
    
    def _correct(self):
        corrections = 0
        for name,trip in self.trips.items():
            i=0
            while trip.size > trip.capacity:
                for student in trip.students:
                    add = 2 if student.jv_coming==1 else 1
                    ith_best_alt = student.preferences[i]
                    if (add+self.trips[ith_best_alt].size) <= self.trips[ith_best_alt].capacity:
                        trip.students.remove(student)
                        self.trips[ith_best_alt].students.append(student)
                        trip.score()
                        self.trips[ith_best_alt].score()
                        corrections+=1
                        break
                i+=1
        self.corrections = corrections
    
    def score(self):
        error = 0
        for trip in self.trips.values():
            trip.score()
            error+=trip.error
        self.error = error


# In[105]:


class Trip(object):
    def __init__(self,name,students,capacity,
                gender_avg,program_avg,country_avg,size=0,error=0,
                gender_dist=None,program_dist=None,country_dist=None):
        self.name = name
        self.students = students
        self.capacity = capacity
        
        self.gender_avg = gender_avg
        self.program_avg = program_avg
        self.country_avg = country_avg
        
        self.size = size
        self.error = error
        
        self.gender_dist = gender_dist
        self.program_dist = program_dist
        self.country_dist = country_dist
        
        self.score()
    
    def score(self):
        size = 0
        gender = 0
        program = 0
        country = 0
        for student in self.students:
            if student.jv_coming==1:
                size+=2
            else:
                size+=1
            if student.gender=='F':
                gender+=1
            if student.program=='2YMBA':
                program+=1
            if student.country=='UNITED STATES':
                country+=1
        
        self.size = size
        self.gender_dist = (gender/size)
        self.program_dist = (program/size)
        self.country_dist = (country/size)
        self.error = (
            abs(self.gender_dist - self.gender_avg)
            + abs(self.program_dist - self.program_avg)
            + abs(self.country_dist - self.country_avg)
        )


# In[106]:


class Student(object):
    def __init__(self,net_id,gender,program,dob,
                country,jv_coming,votes,preferences=None):
        self.net_id = net_id
        self.gender = gender
        self.dob = dob
        self.program = program
        self.country = country
        self.jv_coming = jv_coming
        self.votes = votes
        self.preferences = preferences


# In[ ]:





# In[108]:


start = datetime.datetime.now()

kwest = Kwest(trip_capacity=20)
kwest.setup(weight='exponential')
kwest.predict(weight='exponential',preference='stated')
kwest.match(runs=100)
kwest.pick(preference='match')

print('Runtime:',ceil((datetime.datetime.now()-start).total_seconds()),'seconds')


# In[ ]:





# In[ ]:





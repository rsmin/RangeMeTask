# This file is the answer sheet for questions of stage 1
# SQL - A lot of our data lives in SQL databases and so you will need to be comfortable using SQL. Please answer the questions below and include the SQL statements for each one in your report.

import pandas as pd
import general_functions as gf

# load data into pandas dataframe
_DF = gf.data_loader()


# Task 1, What is the average recommended retail price (RRP) for suppliers who have more than 1 brand?
_Q1 = '''select supplier_rrp_average 
         from _DF 
         where supplier_brand_count>1'''
gf.exec_sql(_Q1, locals(), 1)


# Task 2, What is the total number of shares by suppliers who have been trading for less than 6 years?
_Q2 = '''select sum(supplier_shares_count) 
         from (
             select supplier_shares_count, 
                     CASE
                         WHEN company_years_trading="0-1" THEN 1
                         WHEN company_years_trading="1-2" THEN 2
                         WHEN company_years_trading="3-5" THEN 5
                         WHEN company_years_trading="6-10" THEN 10
                         WHEN company_years_trading="10-15" THEN 15
                         WHEN company_years_trading="15+" THEN 16
                         ELSE 0
                     END AS max_trading_years
             from _DF
             where max_trading_years>0 and max_trading_years<6) T'''
gf.exec_sql(_Q2, locals(), 2)


# Task 3, How many Premium (upgraded) suppliers are there per a state. Order by number of suppliers descending
_Q3 = '''select company_state, sum(upgraded) as total_upgraded 
         from _DF
         group by company_state 
         order by total_upgraded desc'''
gf.exec_sql(_Q3, locals(), 3)


# Task 4, What percentage of suppliers have upgraded and have a buyer popularity score > 50?
_Q4 = '''select count(distinct company_id)*100.0/
                (select count(distinct company_id) 
                 from _DF) as percentage
         from _DF
         where upgraded=1 and buyers_popularity_score>50
'''
gf.exec_sql(_Q4, locals(), 4)
# README

Frino Jais | William Aman | Sri Padmini Jayanti | Minhajul Abadeen  | CSC4996 | June 29 2020

## Prerequisite Programs:
* IDE: [Jetbrains PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows)
* RDBMS: [Oracle MySQL Workbench](https://dev.mysql.com/downloads/workbench/)
* Visualization: [Microsoft Power BI Desktop](https://powerbi.microsoft.com/en-us/desktop/)

## Part 1: MySQL Workbench

1. Launch MySQL Workbench

2. Creat a new connection:  
	* Connection Name: localhost  
	* Hostname: 127.0.0.1    
	* Port: 3306  
	* Username: root  
	* Password: password  
	* (or edit connection string in: -> gmfintech2019 -> FinsterTab -> F2019 -> dbEngine.py)

3. Create schema named 'gmfsp_db' in MySQL Workbench

4. Copy SQL script from folder:  
	* GM-Senior-Capstone-Project-SS2020 -> FinsterTab -> W2020 -> SQL -> SQL CREATE TABLE SCRIPTS MYSQL -> CREATE_DATABASE_TABLES_MYSQL.sql  
	
5. Run script in MySQL Workbench

## Part 2: Python

1. Create a folder named 'gmfintech' at a convenient location on your machine. 

2. Open Command Prompt and navigate to the directory of this 'gmfintech' folder.
	* An easy way to do this is:
		1. Open File Explorer
		2. Navigate to the location of the 'gmfintech' folder
		3. Open the 'gmfintech' folder
		4. Click on the directory path in the Address Bar
		5. Copy this
		6. Return to Command Prompt
		7. Type 'cd '
		8. Paste the address
		9. Press enter

3. Clone GitHub Repository in 'gmfintech' folder by executing the following command:  
	* git clone https://github.com/frinojais/GM-Senior-Capstone-Project-SS2020.git
	* (The Command Prompt should look like this: 
		C:\Users\Frino\Desktop\gmfintech>git clone https://github.com/frinojais/GM-Senior-Capstone-Project-SS2020.git [Press Enter])
					
4. Add Python Interpreter (3.7+)

5. Add all dependencies
	* xgboost 		(PIP script: pip install xgboost)  
	* sqlalchemy 		(PIP script: pip install sqlalchemy) 
	* pandas_datareader 	(PIP script: pip install pandas-datareader)   
	* stockstats 		(PIP script: pip install stockstats)   
	* statsmodels 		(PIP script: pip install statsmodels)   
	* sklearn 		(PIP script: pip install sklearn)  
	* quandl 		(PIP script: pip install quandl) 
	* fredapi 		(PIP script: pip install fredpi) 
	* pytest 		(PIP script: pip install pytest) 
	* pymysql		(PIP script: pip install pymysql) 
	* pyodbc		(PIP script: pip install pyodbc)
	* matplotlib		(PIP script: pip install matplotlib)
	* keras			(PIP script: pip install keras)
	* tensorflow		(PIP script: pip install tensorflow)
	* holidays		(PIP script: pip install holidays)

6. Run DataMain.py in PyCharm


## Part 3: Power BI

1. Open Power BI Desktop

2. Select Get data from the Home tab

3. Search for and select MySQL database

4. Type in the database information (Server: localhost, Database: gmfsp_db)

5. Type in the credentials (User: root, Password: password)

6. Cancel the Navigator window

7. Import the report template using the Import button in the File tab (GM-Senior-Capstone-Project-SS2020 -> FinsterTab -> W2020 -> Power BI -> GM_FinTech_Application.pbit)



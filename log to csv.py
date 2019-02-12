import re

#Using RegEx to filter through .log file, give each company id an array representing number of logins and write to .csv
pattern = re.compile(r'(\d{6}) logged in (\d) times on day (\d)\n')
login_data = {}

try:
    with open('engagement_report.log', 'r') as f:
        contents = f.read()
        matches = pattern.finditer(contents)
except Exception as e:
    print(e)

for match in matches:
    company_id = int(match.group(1))
    logged_in = int(match.group(2))
    day = int(match.group(3))

    login_data[company_id] = login_data.get(company_id,[0, 0, 0, 0, 0, 0, 0])
    login_data[company_id][day-1] = logged_in

with open('7_features_data.csv', 'w') as f:
    columnTitleRow = "Company, Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7\n"
    f.write(columnTitleRow)

    for company_id in login_data:
        row=[str(company_id)]
        for i in range(7):
            row.append(str(login_data[company_id][i]))

        values = ', '.join(row) + '\n'
        f.write(values)


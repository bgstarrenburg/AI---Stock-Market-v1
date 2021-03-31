import classes

a = input('Would you like to refetch data?\n[1]: yes\n[2]: No\n')
while a not in ['1', '2']:
    a = input('Please input a correct value.\n')

if a == '1':
    classes.fetchData()

classes.main()

with open(r'C:\Users\jotab\Documents\Github\NPF\requirements.txt') as req_file, open(r'C:\Users\jotab\Documents\Github\NPF\requirements2.txt', 'a') as f:
    for line in req_file:
        new_line = line.split(' ', 1)[0]
        if not (new_line.endswith('\n')):
            new_line += '\n' 
        if new_line != '\x00\n':
            f.write(new_line)
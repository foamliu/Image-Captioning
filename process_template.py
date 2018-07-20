# -*- coding: utf-8 -*-

if __name__ == '__main__':
    with open('demo.txt', 'r', encoding="utf-8") as file:
        demo = file.readlines()

    with open('README.template', 'r', encoding="utf-8") as file:
        template = file.readlines()

    template = ''.join(template)

    for i in range(20):
        template = template.replace('[{}]'.format(i), demo[i].strip())

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)

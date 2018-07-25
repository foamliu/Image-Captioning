# -*- coding: utf-8 -*-

if __name__ == '__main__':
    with open('demo.txt', 'r', encoding="utf-8") as file:
        demo = file.readlines()

    with open('beam.txt', 'r', encoding="utf-8") as file:
        beam = file.readlines()

    with open('README.template', 'r', encoding="utf-8") as file:
        template = file.readlines()

    template = ''.join(template)

    for i in range(20):
        template = template.replace('[{}]'.format(i), demo[i].strip())

    for i in range(0, 10):
        beam_data = [line.strip() for line in beam[i * 4:(i + 1) * 4]]
        beam_text = '<br>'.join(beam_data)
        template = template.replace('({})'.format(i), beam_text)

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)

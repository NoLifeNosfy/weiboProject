from QA import QA




def main():
    qa = QA()
    # print(qa.get_response("What is a ball"))
    print(qa.chat("My Name is Hamburger"))
    print(qa.chat("what is my name?"))


if __name__ == '__main__':
    main()



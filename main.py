from QA import QA




def main():
    qa = QA()
    print(qa.get_response("Show me how mathemetical induction works?"))
    print(qa.get_response("who are you"))

    # print(qa.chat("My Name is Hamburger"))
    # print(qa.chat("what is my name?"))

if __name__ == '__main__':
    main()



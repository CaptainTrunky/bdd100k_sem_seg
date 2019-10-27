#!/usr/bin/env python3


def main():
    import pymongo

    client = pymongo.MongoClient()
    client.drop_database('sacred')


if __name__ == '__main__':
    main()

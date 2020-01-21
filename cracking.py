from pprint import pprint

def uniqueNoData():
    test = "nodata"

    arr = [0] * 26
    for s in test:
        key = ord(s.lower()) - ord('a')
        print(key)
        if arr[key] > 0:
            return False
        else:
            arr[key] += 1
    return True

def rotate90(img):


    imgCopy = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],]

    pprint(img)
    print(imgCopy)


    for i in range(len(img)-1,-1,-1):
        for j in range(0,len(img)):
            # imgCopy[j][i] = img[len(img)-j-1][j]
            imgCopy[len(img)-i-1][j] = img[j][i]
    pprint(imgCopy)
    return imgCopy

def bruteZeros(img):
    pprint(img)
    res = []
    isZero = False
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 0:
                res.append((i,j))
                isZero = True
                break

        if isZero:
            continue
    print(res)
    #unique
    # make x 0
    resRow = set([r[0] for r in res])
    resCol = set([r[1] for r in res])
    for i in range(len(img)):
        for j in range(len(img[0])):
            if i in resRow or j in resCol:
                img[i][j] = 0

    pprint(img)
    # resCol = set([r[1] for r in res])
    # for j in range(len(img[0])):
    #     if j in resRow or j in resCol:
    #         for i in range(len(img)):
    #             img[i][j] = 0
    #
    # pprint(img)

class Node(object):

    def __init__(self, dt):
        self.next = None
        self.data = dt



class LinkedList(object):

    def __init__(self, dt):
        self.head = Node(dt)

    def add(self, dt):
        node = self.head
        while node.next is not None:
            # node.next = Node(dt)
            # node = node.next.next
            node = node.next
        node.next = Node(dt)
        # if self.head is not None:
        #     self.head.next = Node(dt)

    def Head(self):
        return self.head

    def printList(self):
        res = []
        node = self.head
        while node is not None:
            res.append(node.data)
            node = node.next
        print(res)

    def printNFromLast(self, n):
        # move first pointer n spaces
        node1 = self.head
        node2 = self.head
        i = 0
        while node1 is not None and i < n:
            node1 = node1.next
            i += 1

        if node1 is None:
            return node2

        while node1 is not None:
            node1 = node1.next
            node2 = node2.next

        return node2

    def deleteNode(self, node):
        prevNode = node
        while node is not None:
            nextNode = node.next
            if nextNode is not None:
                node.data = nextNode.data
            else:
                break
            prevNode = node
            node = nextNode

        if prevNode is not None:
            # print('enter',hex(id(prevNode)))
            prevNode.next = None


def addLinkedLists(l1, l2):
    node1 = l1.head
    node2 = l2.head
    res = None
    carry = 0
    while node1 is not None and node2 is not None:
        s = node1.data + node2.data + carry
        carry = int(s / 10) if s >= 10 else 0
        s = s % 10 if s >= 10 else s
        if not res:
            res = LinkedList(s)
        else:
            res.add(s)
        node1 = node1.next
        node2 = node2.next

    while node1 is not None:
        s = node1.data + carry

        carry = int(s / 10) if s >= 10 else 0
        s = s % 10 if s >= 10 else s
        res.add(s)
        node1 = node2.next

    while node2 is not None:
        s = node2.data + carry
        carry = int(s / 10) if s >= 10 else 0
        s = s % 10 if s >= 10 else s
        res.add(s)
        node2 = node2.next

    if carry:
        res.add(carry)

    return res


def pracTrie():
    words = ['hackerrank', 'hackerearth']

    trie={}
    for w in words:
        t=trie
        for c in w:
            if c not in t:
                t[c]={}
            t=t[c]
        t['#']='#'
        print(trie)
    print(trie)

if __name__ == "__main__":
    # print(uniqueNoData())
    # img = [['a','b','c','d'],['a','b','c','d'],['a','b','c','d'],['a','b','c','d'],]
    # img = rotate90(img)
    # rotate90(img)
    # img = [list(range(5,10)),list(range(5)),list(range(10,15))]
    # bruteZeros(img)
    # linkedList = LinkedList(0)
    # for i in range(1,9):
    #     linkedList.add(i)
    # linkedList.printList()
    # resNode = linkedList.printNFromLast(5)
    # print(resNode.data)
    # linkedList.deleteNode(resNode)
    # linkedList.printList()
    # l1 = LinkedList(9)
    # l1.add(9)
    # l1.add(9)
    # l2 = LinkedList(9)
    # l2.add(9)
    # l2.add(9)
    # res = addLinkedLists(l1,l2)
    # res.printList()
    # linkedList.head.data
    # node = linkedList.head
    pracTrie()


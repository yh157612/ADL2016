class TreeNode(object):

    def __init__(self, content=''):
        self.content = content
        self.children = []

    def print_subtree(self):
        print('(', end='')
        print(self.content, end='')
        for child in self.children:
            child.print_subtree()
        print(')', end='')

    def binarize(self):
        while len(self.children) == 1:
            self.content = self.children[0].content
            self.children = self.children[0].children
        index = self.content.find(' ')
        if index >= 0:
            self.content = self.content[index + 1:]
        else:
            self.content = ''
        if len(self.children) > 2:
            node = TreeNode()
            node.children = self.children[1:]
            self.children = [self.children[0], node]
        for child in self.children:
            child.binarize()

    def words(self):
        if not self.children:
            return [self.content]
        return self.children[0].words() + self.children[1].words()


def read_until_bracket(f):
    delim_list = ['', '(', ')']
    c = f.read(1)
    result = ''
    while c not in delim_list:
        result += c
        c = f.read(1)
    return result, c


def parse(f):
    content, c = read_until_bracket(f)
    node = TreeNode(content.strip())
    assert c != ''
    while c == '(':
        node.children.append(parse(f))  # Recursive call
        _, c = read_until_bracket(f)
    return node


def read_one_tree(f, binarize=True):
    _, c = read_until_bracket(f)
    if c == '':
        return None
    assert c == '('
    root = parse(f)
    if binarize:
        root.binarize()
    return root


def load_trees(path):
    trees = []
    with open(path) as f:
        while True:
            root = read_one_tree(f)
            if root is None:
                break
            trees.append(root)
    return trees

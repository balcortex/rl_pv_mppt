class Add:
    def execute(self, x, y):
        self.x = x
        self.y = y
        return x + y


class Add2(Add):
    def execute(self, x, y):
        a = super().execute(x, y)
        # return self.x + self.y + 10
        return a + 10


add = Add()
add2 = Add2()


print(add.execute(4, 3))
print(add2.execute(20, 30))

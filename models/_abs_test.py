

# NOTE - child class attributes accessing inside parent class when the parent is an abstract class means there is no object or instance exist from it
# https://stackoverflow.com/questions/25062114/calling-child-class-method-from-parent-class-file-in-python

class parent:
    def __init__(self, **kwargs):
        self.id = kwargs["id"]
        if self.__class__.__name__ == "child": 
            print(self) # child __repr__
            print(self.where())
        else:
            print(self) # parent __repr__
            print(self.some_method())
    
    def __repr__(self):
        return "parent class repr"
    
    def some_method(self):
        return "inside the parent class which is not an abstract class"



class child(parent):
    def __init__(self):
        self.msg = "inside child class"
        self.args = {"name": "wildonion", "id": "999367704"}
        super().__init__(**self.args)
    
    def __repr__(self):
        return "child class repr"

    def where(self):
        return self.msg


c = child()
print(".....parent building....")
p = parent(**c.args)

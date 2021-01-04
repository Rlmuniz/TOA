class Runway:

    def __init__(self, length: float, clearway: float = 0.0, stopway: float = 0.0):
        self._length = length
        self.clearway = clearway
        self.stopway = stopway

    @property
    def tora(self):
        return self._length

    @property
    def clearway(self):
        return self._clearway

    @clearway.setter
    def clearway(self, value):
        max_clear_way_length = 0.5 * self._length
        if value <= 0:
            self._clearway = 0
        elif value > max_clear_way_length:
            self._clearway = max_clear_way_length
        else:
            self._clearway = value

    @property
    def toda(self):
        return self._length + self.clearway

    @property
    def asda(self):
        return self._length + self.stopway

if __name__ == '__main__':
    r1 = Runway(3000, 1000, 1000)
    print(r1.tora, r1.toda, r1.asda, r1.clearway, r1.stopway)
    r2 = Runway(3000, 1600, 1600)
    print(r2.tora, r2.toda, r2.asda, r2.clearway, r2.stopway)
    r2.clearway = 1400
    print(r2.toda)
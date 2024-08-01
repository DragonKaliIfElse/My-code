class EstruturaDoJogo:
	def __init__(self, valorDaRoleta):
		firstColun = [1,4,7,10,13,16,19,22,25,28,31,34]
		seconColun = [2,5,8,11,14,17,20,23,26,29,32,35]
		thirdColun = [3,6,9,12,15,18,21,24,27,30,33,36]
		zero = [0]
		firstDozen = [1,2,3,4,5,6,7,8,9,10,11,12]
		seconDozen = [13,14,15,16,17,18,19,20,21,22,23,24]
		thirdDozen = [25,26,27,28,29,30,31,32,33,34,35,36]
		blackNumbers=[]
		i=2
		while i<=10:
			blackNumbers.append(i)
			i+=2
		i=11
		while i<=17:
			blackNumbers.append(i)
			i+=2
		i=20
		while i<=28:
			blackNumbers.append(i)
			i+=2
		i=29
		while i<=35:
			blackNumbers.append(i)
			i+=2
		redNumbers=[]
		i=1
		while i<=9:
			redNumbers.append(i)
			i+=2
		i=12
		while i<=18:
			redNumbers.append(i)
			i+=2
		i=19
		while i<=27:
			redNumbers.append(i)
			i+=2
		i=30
		while i<=36:
			redNumbers.append(i)
			i+=2
		i=1
		oneUntilEighteen=[]
		while i <=18:
			oneUntilEighteen.append(i)
			i+=1
		nineteenUntilThirtySix=[]
		while i <=36:
			nineteenUntilThirtySix.append(i)
			i+=1
		i=2
		even=[]
		while i <=36:
			even.append(i)
			i+=2
		i=1
		odd=[]
		while i <=35:
			odd.append(i)
			i+=2
		i=0
		self.odd = odd
		self.even = even
		self.blackNumbers = blackNumbers
		self.redNumbers = redNumbers
		self.nineteenUntilThirtySix = nineteenUntilThirtySix
		self.oneUntilEighteen = oneUntilEighteen
		self.firstColun = firstColun
		self.seconColun = seconColun
		self.thirdColun = thirdColun
		self.firstDozen = firstDozen
		self.seconDozen = seconDozen
		self.thirdDozen = thirdDozen
		self.zero = zero
		self.valorDaRoleta = int(valorDaRoleta)

	def Verifica(self, array):
		booleano = False
		for i in array:
			if i == self.valorDaRoleta:
				booleano = True
		return booleano

	def firstColun_(self):
		booleano = self.Verifica(self.firstColun)
		return booleano

	def seconColun_(self):
		booleano = self.Verifica(self.seconColun)
		return booleano

	def thirdColun_(self):
		booleano = self.Verifica(self.thirdColun)
		return booleano

	def firstDozen_(self):
		booleano = self.Verifica(self.firstDozen)
		return booleano

	def seconDozen_(self):
		booleano = self.Verifica(self.seconDozen)
		return booleano
	def thirdDozen_(self):
		booleano = self.Verifica(self.thirdDozen)
		return booleano

	def zero_(self):
		booleano = self.Verifica(self.zero)
		return booleano

	def oneUntilEighteen_(self):
		booleano = self.Verifica(self.oneUntilEighteen)
		return booleano

	def nineteenUntilThirtySix_(self):
		booleano = self.Verifica(self.nineteenUntilThirtySix)
		return booleano

	def even_(self):
		booleano = self.Verifica(self.even)
		return booleano

	def odd_(self):
		booleano = self.Verifica(self.odd)
		return booleano

	def blackNumbers_(self):
		booleano = self.Verifica(self.blackNumbers)
		return booleano

	def redNumbers_(self):
		booleano = self.Verifica(self.redNumbers)
		return booleano

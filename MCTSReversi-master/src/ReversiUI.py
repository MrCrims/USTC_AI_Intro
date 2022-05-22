import sys
import mcts_reversi as rvs

import tkinter as Tk


class ReversiBoard(Tk.Canvas): 
	cell_size = 46
	margin = 5

	board = rvs.getInitialBoard()
	validBoard    = True
	isPayerTurn   = True

	objids = []

	def __init__(self, master):
		cwidth = rvs.BOARD_SIZE * self.cell_size

		Tk.Canvas.__init__(self, master, relief=Tk.RAISED, bd=4, bg='white', width=cwidth, height=cwidth)

		self.bind("<1>", self.put_stones)

		for i in range(rvs.BOARD_SIZE):
			for j in range(rvs.BOARD_SIZE):
				bcolor = "#699C69"
				x0 = i*self.cell_size + self.margin
				y0 = j*self.cell_size + self.margin
				self.create_rectangle(x0, y0, x0+self.cell_size, y0+self.cell_size, fill=bcolor, width=1)		
		self.refresh()

	def put_stones(self, event):
		if self.validBoard == False:
			self.validBoard = True
			self.board = rvs.getInitialBoard()
			self.isPayerTurn = True

			for numid in self.objids:
				self.delete(numid)
			self.objids = []
			self.refresh()
			print("refreshed")

			return

		if not(self.isPayerTurn):
			return

		cx = self.canvasx(event.x)
		cy = self.canvasy(event.y)

		i = int(cx/self.cell_size)
		j = int(cy/self.cell_size)
		if (0<=i<rvs.BOARD_SIZE and 0<=j<rvs.BOARD_SIZE):
			print("pointed: (", i, ",", j, ")")

		if self.board[i][j] != 0 or rvs.updateBoard(self.board, rvs.PLAYER_NUM, i, j, checkonly=True) == 0:
			return

		rvs.updateBoard(self.board, rvs.PLAYER_NUM, i, j)
		self.refresh()
		isPayerTurn = False
		self.after(100, self.ai)

	def ai(self):
		while True:
			player_possibility = len(rvs.checkPlacablePositions(self.board, rvs.PLAYER_NUM))
			mcts_possibility   = len(rvs.checkPlacablePositions(self.board, rvs.MCTS_NUM))

			if mcts_possibility == 0:
				break
			
			stone_pos = rvs.mctsNextPosition(self.board)
			print("MCTS's answer: ", stone_pos)
			rvs.updateBoard(self.board, rvs.MCTS_NUM, stone_pos[0], stone_pos[1])
			self.refresh()

			player_possibility = len(rvs.checkPlacablePositions(self.board, rvs.PLAYER_NUM))
			mcts_possibility   = len(rvs.checkPlacablePositions(self.board, rvs.MCTS_NUM))

			if mcts_possibility == 0 or player_possibility > 0:
				break

		if player_possibility == 0 and mcts_possibility == 0:
			self.showResult()
			self.validBoard = False

		self.isPayerTurn = True

	def showResult(self):
		player_stone = rvs.countStones(self.board, rvs.PLAYER_NUM)
		mcts_stone   = rvs.countStones(self.board, rvs.MCTS_NUM)
		
		if player_stone > mcts_stone:
			msg = "You won"
		elif player_stone == mcts_stone:
			msg = "Draw"
		else:
			msg = "You lose"
		
		idnum = self.create_text('2c', '1.5c', text=msg, font=('Helvetica', '20', 'bold'), fill="#FF0000")
		self.objids.append( idnum )

	def refresh(self):
		for i in range(rvs.BOARD_SIZE):
			for j in range(rvs.BOARD_SIZE):
				x0 = i*self.cell_size + self.margin
				y0 = j*self.cell_size + self.margin

				if self.board[i][j] == 0:
					continue
				if self.board[i][j] == rvs.PLAYER_NUM :
					bcolor = "#000000"
				if self.board[i][j] == rvs.MCTS_NUM:
					bcolor = "#ffffff"
				idnum = self.create_oval(x0+2, y0+2, x0+self.cell_size-2, y0+self.cell_size-2, fill=bcolor, width=0)
				self.objids.append( idnum )
		

class Reversi(Tk.Frame):

	def __init__(self, master=None):
		Tk.Frame.__init__(self, master)
		self.master.title("Reversi")


		# title
		l_title = Tk.Label(self, text='Reversi', font=('Times', '24', ('italic', 'bold')), fg='#191970', bg='#EEE8AA', width=12)

		l_title.pack(padx=10, pady=10)
		
		self.f_board = ReversiBoard(self)
		self.f_board.pack(padx=10, pady=10)



if __name__ == '__main__':

	app = Reversi()
	app.pack()
	app.mainloop()

	


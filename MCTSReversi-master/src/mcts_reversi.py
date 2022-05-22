
import random
import math
import time



BOARD_SIZE = 8

PLAYER_NUM = 2
MCTS_NUM   = 1

MAX_THINK_TIME = 3



# Return a new initialized board

def getInitialBoard():
	board = {}

	for i in range(0,BOARD_SIZE):
		board[i] = {}

		for j in range(0,BOARD_SIZE):
			board[i][j] = 0

	board[BOARD_SIZE/2 -1][BOARD_SIZE/2 -1] = MCTS_NUM
	board[BOARD_SIZE/2   ][BOARD_SIZE/2   ] = MCTS_NUM

	board[BOARD_SIZE/2 -1][BOARD_SIZE/2   ] = PLAYER_NUM
	board[BOARD_SIZE/2   ][BOARD_SIZE/2 -1] = PLAYER_NUM

	return board


# 复制棋盘

def copyBoard(dest_board, src_board):

	for i in range(0, BOARD_SIZE):
		for j in range(0, BOARD_SIZE):
			dest_board[i][j] = src_board[i][j]



# 统计特定棋子的数量

def countStones(board, turn):
	stones = 0
	
	for i in range(0, BOARD_SIZE):
		for j in range(0, BOARD_SIZE):
			if board[i][j] == turn:
				stones += 1

	return stones


# 统计可以落子的位置

def checkPlacablePositions(board, turn):

	placable_positions = []

	for i in range(0, BOARD_SIZE):
		for j in range(0, BOARD_SIZE):
			if board[i][j] != 0:
				continue
			if updateBoard(board, turn, i, j, checkonly=True) > 0:
				placable_positions.append( (i,j) )
	return placable_positions



# 检查将棋子在当前状态下（board和turn）放置在（i,j）处是否合理，如果可以放置则更新棋盘

def updateBoard(board, turn, i, j, checkonly=False):
	
	if not(checkonly):
		board[i][j] = turn

	reversed_stone = 0


	flag = False
	for i2 in range(i+1, BOARD_SIZE):
		if board[i2][j] == 0 :
			break
		if board[i2][j] == turn:
			flag = True
			break
	if flag:
		for i3 in range(i+1, i2):
			if not(checkonly):
				board[i3][j] = turn
			reversed_stone += 1

	flag = False
	for i2 in reversed(range(0, i)):
		if board[i2][j] == 0 :
			break
		if board[i2][j] == turn:
			flag = True
			break
	if flag:
		for i3 in reversed(range(i2+1, i)):
			if not(checkonly):
				board[i3][j] = turn
			reversed_stone += 1


	flag = False
	for j2 in range(j+1, BOARD_SIZE):
		if board[i][j2] == 0 :
			break
		if board[i][j2] == turn:
			flag = True
			break
	if flag:
		for j3 in range(j+1, j2):
			if not(checkonly):
				board[i][j3] = turn
			reversed_stone += 1

	flag = False
	for j2 in reversed(range(0, j)):
		if board[i][j2] == 0 :
			break
		if board[i][j2] == turn:
			flag = True
			break
	if flag:
		for j3 in reversed(range(j2+1, j)):
			if not(checkonly):
				board[i][j3] = turn
			reversed_stone += 1


	flag = False
	for m in range(1, min(BOARD_SIZE-i, BOARD_SIZE-j)):
		if board[i+m][j+m] == 0 :
			break
		if board[i+m][j+m] == turn:
			flag = True
			break
	if flag:
		for m3 in range(1, m):
			if not(checkonly):
				board[i+m3][j+m3] = turn
			reversed_stone += 1

	flag = False
	for m in range(1, min(i,j)+1):
		if board[i-m][j-m] == 0 :
			break
		if board[i-m][j-m] == turn:
			flag = True
			break
	if flag:
		for m3 in range(1, m):
			if not(checkonly):
				board[i-m3][j-m3] = turn
			reversed_stone += 1


	flag = False
	for m in range(1, min(i+1, BOARD_SIZE-j)):
		if board[i-m][j+m] == 0 :
			break
		if board[i-m][j+m] == turn:
			flag = True
			break
	if flag:
		for m3 in range(1, m):
			if not(checkonly):
				board[i-m3][j+m3] = turn
			reversed_stone += 1

	flag = False
	for m in range(1, min(BOARD_SIZE-i,j+1)):
		if board[i+m][j-m] == 0 :
			break
		if board[i+m][j-m] == turn:
			flag = True
			break
	if flag:
		for m3 in range(1, m):
			if not(checkonly):
				board[i+m3][j-m3] = turn
			reversed_stone += 1


	return reversed_stone




# 用MCTS来实现对下一个落子点的预测

def mctsNextPosition(board):

	# UCB1计算（节点选择使用）
	def calc_ucb1( node_tuple, t, cval ):
		name, nplayout, reward, childrens = node_tuple

		if nplayout == 0:
			nplayout = 0.00000000001

		if t == 0:
			t = 1

		return (reward / nplayout) + cval * math.sqrt( 2*math.log( t ) / nplayout )

	# 递归向下模拟落子
	def find_playout( brd, turn, depth = 0):
		def eval_board( brd ):
			player_stone = countStones(brd, PLAYER_NUM)
			mcts_stone   = countStones(brd, MCTS_NUM)

			if mcts_stone > player_stone:
				return True
			return False


		# 递归深度控制
		if depth > 32:
			return eval_board( brd )

		turn_positions = checkPlacablePositions(brd, turn)

		# 检查当前棋手能否下棋，不能则交换棋权
		if len(turn_positions) == 0:
			if turn == MCTS_NUM:
				neg_turn = PLAYER_NUM
			else:
				neg_turn = MCTS_NUM

			neg_turn_positions = checkPlacablePositions(brd, neg_turn)
			
			if len(neg_turn_positions) == 0:
				# 大家都下不了时就游戏结束，返回棋盘统计结果
				return eval_board( brd )
			else:
				# 交换棋权
				turn = neg_turn
				turn_positions = neg_turn_positions
		
		# 随机落子然后检查能否更新
		ijpair = turn_positions[ random.randrange(0, len(turn_positions)) ]
		updateBoard(brd, turn, ijpair[0], ijpair[1])

		# 交换棋权
		if turn == MCTS_NUM:
			turn = PLAYER_NUM
		else:
			turn = MCTS_NUM

		return find_playout( brd, turn, depth=depth+1)

	# 扩展节点
	def expand_node(brd, turn):
		positions = checkPlacablePositions(brd, turn)
		result = []
		
		for ijpair in positions:
			result.append( (ijpair, 0, 0, []) )

		return result

	# 获得一条优路径
	def find_path_default_policy( root, total_playout ):
		current_path      = []
		current_childrens = root

		parent_playout = total_playout

		isMCTSTurn = True

		while True:
			if len(current_childrens) == 0:
				break

			maxidxlist = [0]
			cidx       = 0
			if isMCTSTurn:
				maxval = -1
			else:
				maxval = 2

			for n_tuple in current_childrens:
				t_ijpair, t_nplayout, t_reward, t_childrens = n_tuple

				# In MCTS's turn, the node with the largest value is selected.
				# In the other's turn, the node with the least value is selected.

				if isMCTSTurn:
					cval = calc_ucb1( n_tuple, parent_playout, 0.1 )

					if cval >= maxval:
						if cval == maxval:
							maxidxlist.append( cidx )
						else:
							maxidxlist = [ cidx ]
							maxval     = cval
				else:
					cval = calc_ucb1( n_tuple, parent_playout, -0.1 )

					if cval <= maxval:
						if cval == maxval:
							maxidxlist.append( cidx )
						else:
							maxidxlist = [ cidx ]
							maxval     = cval

				cidx += 1

			# 随机选择
			maxidx = maxidxlist[ random.randrange(0, len(maxidxlist)) ]
			t_ijpair, t_nplayout, t_reward, t_childrens = current_childrens[maxidx]

			current_path.append( t_ijpair )
			parent_playout = t_nplayout
			current_childrens = t_childrens

			isMCTSTurn = not(isMCTSTurn)

		return current_path


	root = expand_node(board, MCTS_NUM)
	current_board = getInitialBoard()
	current_board2 = getInitialBoard()

	start_time = time.time()


	t0 = time.time()
	for loop in range(0, 5000):

		# 检查思考时间限制
		if (time.time() - start_time) >= MAX_THINK_TIME:
			break

		current_path = find_path_default_policy( root, loop )

		# current_path contains a list of positions to be placed stones.
		# Following lines places stones according to the list.
		# 
		# Note that the turn changes alternately.
		# Currently, a node corresponding to the situation requesting "pass" has no child nodes.
		# Thus it is not necessary to consider a player placing multiple stones at once.

		copyBoard(current_board, board)
		turn = MCTS_NUM
		for ijpair in current_path:
			updateBoard(current_board, turn, ijpair[0], ijpair[1])
			if turn == MCTS_NUM:
				turn = PLAYER_NUM
			else:
				turn = MCTS_NUM



		# 复制更新后的board到board2作为备份并判断是否胜利

		copyBoard(current_board2, current_board)
		isWon = find_playout( current_board2, turn)


		# 反向传播

		current_childrens = root

		for ijpair in current_path:
			idx = 0
			for n_tuple in current_childrens:
				t_ijpair, t_nplayout, t_reward, t_childrens = n_tuple
				if ijpair[0] == t_ijpair[0] and ijpair[1] == t_ijpair[1]:
					break
				idx += 1

			if ijpair[0] == t_ijpair[0] and ijpair[1] == t_ijpair[1]:
				t_nplayout += 1
				if isWon:
					t_reward   += 1
				
				if t_nplayout >= 5 and len(t_childrens) == 0:
					t_childrens = expand_node(current_board, turn)

				current_childrens[idx] = (t_ijpair, t_nplayout, t_reward, t_childrens)
			else:
				print("failed")

			current_childrens = t_childrens


	tf = time.time()
	print("loop time: ",tf-t0)

	max_avg_reward = -1
	result_ij_pair = (0,0)

	for n_tuple in root:
		t_ijpair, t_nplayout, t_reward, t_childrens = n_tuple

		if (t_nplayout > 0) and (t_reward / t_nplayout > max_avg_reward):
			result_ij_pair = t_ijpair
			max_avg_reward = t_reward / t_nplayout
	
	return result_ij_pair







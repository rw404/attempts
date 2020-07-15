extends Area2D

#Variables for all body blocks:
	#array of directions
var directions = []
	#array of positions
var pos_array = []
	#current direction
var cur_dir = Vector2.ZERO
	#signal that snake bumped into its body
signal touch

func _physics_process(delta):
	#each time if directions is not empty diredction updates(it is the previous
	#direction of next block), when previous position is reached arrays updates
	if(directions.size() > 0):
		if(position == pos_array[0]):
			cur_dir = directions[0]
			#updating arrays(removing the first elements)
			remove_from_tail()
	#moving
	position += cur_dir/2

func remove_from_tail():
	directions.pop_front()
	pos_array.pop_front()

func add_to_tail(head_pos, direction):
	#updating arrays, adding current position of head element and direction
	pos_array.append(head_pos)
	directions.append(direction)

func _on_Body_area_entered(area):
	#when snake bumped, each previous blocks are being destroying
	if(area.name == "head"):
		emit_signal("touch", self)

extends Area2D

#the same as Body.gd, but snake can bump into its snake and tail is animated
var directions = []
var pos_array = []
var cur_dir = Vector2.ZERO

func _physics_process(delta):
	if(directions.size() > 0):
		if(position == pos_array[0]):
			cur_dir = directions[0]
			remove_from_tail()
	position += cur_dir/2
	
	#animation for each direction
	if(cur_dir == Vector2(1, 0)):
		$tail_anim.play("tail_right")
	elif(cur_dir == Vector2(-1, 0)):
		$tail_anim.play("tail_left")
	elif(cur_dir == Vector2(0, 1)):
		$tail_anim.play("tail_down")
	elif(cur_dir == Vector2(0, -1)):
		$tail_anim.play("tail_up")
	else:
		pass

func remove_from_tail():
	directions.pop_front()
	pos_array.pop_front()

func add_to_tail(head_pos, direction):
	pos_array.append(head_pos)
	directions.append(direction)

func _on_Tail_area_entered(area):
	if(area.name == "head"):
		pass

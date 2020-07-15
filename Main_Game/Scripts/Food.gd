extends Area2D

#signal, that food element was eaten
signal eaten

#function-signal
func _on_Food_area_entered(area):
	
	#if the area which entered food area is snake(head), then food objects
	#can be destroyed and signal "eaten" is on
	if(area.name == "head"):
		queue_free()
		emit_signal("eaten")

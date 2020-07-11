extends Area2D

signal eaten

func _on_Food_area_entered(area):
	if(area.name == "head"):
		queue_free()
		emit_signal("eaten")

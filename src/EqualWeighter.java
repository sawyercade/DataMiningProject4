import java.util.List;

public class EqualWeighter implements ModelWeighter{
    @Override
    public void weight(List<Model> models) {
        int n = models.size();
        for (Model model : models){
            model.setWeight(1.0/n);
        }
    }
}

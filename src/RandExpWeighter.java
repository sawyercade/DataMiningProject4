import helpers.Rand;

import java.util.List;

public class RandExpWeighter implements ModelWeighter {

    @Override
    public void weight(List<Model> models) {
        for (Model model : models){
            model.setWeight(Rand.drawFromStandardExp());
        }
    }
}

package hust.soict.dsai.aims.store;
import java.util.ArrayList;
import hust.soict.dsai.aims.disc.DigitalVideoDisc;

public class Store {
    private ArrayList<DigitalVideoDisc> itemsInStore; 

    public Store() {
        this.itemsInStore = new ArrayList<>();
    }

    public void addDVD(DigitalVideoDisc dvd) {
        if (dvd != null) {
            itemsInStore.add(dvd);
            System.out.println("DVD added to store: " + dvd.getTitle());
        } else {
            System.out.println("Cannot add a null DVD.");
        }
    }

    public void removeDVD(DigitalVideoDisc dvd) {
        if (itemsInStore.contains(dvd)) {
            itemsInStore.remove(dvd);
            System.out.println("DVD removed from store: " + dvd.getTitle());
        } else {
            System.out.println("DVD not found in store: " + (dvd != null ? dvd.getTitle() : "null"));
        }
    }

    public void displayStore() {
        System.out.println("DVDs currently in store:");
        for (DigitalVideoDisc dvd : itemsInStore) {
            System.out.println("- " + dvd.getTitle());
        }
    }
}

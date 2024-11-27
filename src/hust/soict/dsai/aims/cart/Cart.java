package hust.soict.dsai.aims.cart;
import hust.soict.dsai.aims.disc.DigitalVideoDisc;
import java.util.*;

public class Cart {
	private int qtyOrdered = 0;
	public static final int MAX_NUMBER_OF_ORDER = 20;
	private ArrayList<DigitalVideoDisc> itemsOrdered = new ArrayList<>();

    public void addDigitalVideoDisc(DigitalVideoDisc disc) {
        if (itemsOrdered.size() < MAX_NUMBER_OF_ORDER) {
            itemsOrdered.add(disc);
            System.out.println("The DVD is added to the cart.");
        } else {
            System.out.println("The cart is full.");
        }
    }
	
	public void addDigitalVideoDisc(DigitalVideoDisc[] dvdList) {
		for (DigitalVideoDisc dvd : dvdList) {
			addDigitalVideoDisc(dvd);
		}
	}
	
	public void addDigitalVideoDisc(DigitalVideoDisc dvd1, DigitalVideoDisc dvd2) {
		addDigitalVideoDisc(dvd1);
		addDigitalVideoDisc(dvd2);
	}
	public void removeDigitalVideoDisc(DigitalVideoDisc disc) {
	        if (itemsOrdered.remove(disc)) {
	            System.out.println("The DVD is removed.");
	        } else {
	            System.out.println("Cannot find the disc.");
	        }
	    }
	
	public float totalCost() {
	    float totalCost = 0;
	    for (DigitalVideoDisc dvd : itemsOrdered) {
	        totalCost += dvd.getCost();
	    }
	    return totalCost;
	}
	
	public void searchByID(int searchID) {
	    for (DigitalVideoDisc dvd : itemsOrdered) {
	        if (dvd.getId() == searchID) { 
	            System.out.println("Found DVD: " + dvd.getTitle());
	            return;
	        }
	    }
	    System.out.println("No results found.");
	}
	
	public void searchByTitle(String searchTitle) {
	    ArrayList<DigitalVideoDisc> searchResults = new ArrayList<>();
	    for (DigitalVideoDisc dvd : itemsOrdered) {
	        if (dvd.isMatch(searchTitle)) { // Assuming DigitalVideoDisc has an isMatch() method
	            searchResults.add(dvd);
	        }
	    }

	    if (searchResults.isEmpty()) {
	        System.out.println("No results found.");
	    } else {
	        System.out.println("Search results:");
	        for (DigitalVideoDisc dvd : searchResults) {
	            System.out.println("- " + dvd.getTitle());
	        }
	    }
	}
	
	public void printOrder() {
	    for (DigitalVideoDisc dvd : itemsOrdered) {
	        System.out.println(dvd.toString());
	    }
	}
	
    public void detailDisplayCart() {
        System.out.println("***********************CART***********************");
        System.out.println("Ordered Items:");
        for (int i = 0; i < itemsOrdered.size(); i++) {
            System.out.println((i + 1) + ". " + itemsOrdered.get(i).toString());
        }
        System.out.println("Total cost: " + totalCost() + "$");
        System.out.println("***************************************************");
    }
}

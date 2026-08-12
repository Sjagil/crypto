use nautilus_model::{
    data::BookOrder,
    enums::{BookType, OrderSide},
    identifiers::InstrumentId,
    orderbook::OrderBook,
    types::{Price, Quantity},
};

fn parse_arg(args: &[String], index: usize) -> f64 {
    args[index].parse::<f64>().expect("numeric probe argument")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        std::process::exit(2);
    }
    let bid = parse_arg(&args, 1);
    let ask = parse_arg(&args, 2);
    let bid_size = parse_arg(&args, 3);
    let ask_size = parse_arg(&args, 4);
    let instrument_id = InstrumentId::from("BTC-EUR.BITVAVO");
    let mut book = OrderBook::new(instrument_id, BookType::L2_MBP);
    book.add(
        BookOrder::new(OrderSide::Buy, Price::new(bid, 8), Quantity::new(bid_size, 8), 0),
        0,
        1,
        1.into(),
    );
    book.add(
        BookOrder::new(OrderSide::Sell, Price::new(ask, 8), Quantity::new(ask_size, 8), 0),
        0,
        2,
        2.into(),
    );
    println!(
        "{{\"spread\":{},\"midpoint\":{}}}",
        book.spread().expect("spread"),
        book.midpoint().expect("midpoint")
    );
}

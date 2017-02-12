//==================================
// XML読み書き用クラス
//==================================
#pragma once

#ifdef XMLUTILITY_EXPORTS
#define XMLUTILITY_API __declspec(dllexport)
#else
#define XMLUTILITY_API __declspec(dllimport)
#endif

#include<string>
#include<vector>
#include<map>

#include<boost/shared_ptr.hpp>
#include<./boost/uuid/uuid.hpp>

namespace XMLUtility
{
	enum NodeType
	{
		XmlNodeType_None	= 0,
		XmlNodeType_Element	= 1,
		XmlNodeType_Attribute	= 2,
		XmlNodeType_Text	= 3,
		XmlNodeType_CDATA	= 4,
		XmlNodeType_ProcessingInstruction	= 7,
		XmlNodeType_Comment	= 8,
		XmlNodeType_DocumentType	= 10,
		XmlNodeType_Whitespace	= 13,
		XmlNodeType_EndElement	= 15,
		XmlNodeType_XmlDeclaration	= 17,
		_XmlNodeType_Last	= 17
	};


	class XMLUTILITY_API IXmlWriter
	{
	public:
		/** コンストラクタ */
		IXmlWriter(){}
		/** デストラクタ */
		virtual ~IXmlWriter(){}

	public:
		/** 要素を開始 */
		virtual LONG StartElement(const std::string&  name) = 0;
		virtual LONG StartElement(const std::wstring& name) = 0;

		/** 要素を終了 */
		virtual LONG EndElement() = 0;


		/** 要素に文字列を書き込み */
		virtual LONG WriteElement(const std::string&  name, const std::string& value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, const std::wstring& value) = 0;
		/** 要素に文字列を書き込み */
		virtual LONG WriteElement(const std::string&  name, const char value[]) = 0;
		virtual LONG WriteElement(const std::wstring&  name, const WCHAR value[]) = 0;
		/** 要素に整数(32bit)を書き込み */
		virtual LONG WriteElement(const std::string&  name, __int32 value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, __int32 value) = 0;
		/** 要素に整数(64bit)を書き込み */
		virtual LONG WriteElement(const std::string&  name, __int64 value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, __int64 value) = 0;
		/** 要素に符号無し整数を書き込み */
		virtual LONG WriteElement(const std::string&  name, unsigned long value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, unsigned long value) = 0;
		/** 要素に実数を書き込み */
		virtual LONG WriteElement(const std::string&  name, double value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, double value) = 0;
		/** 要素に論理値を書き込み */
		virtual LONG WriteElement(const std::string&  name, bool value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, bool value) = 0;
		/** 要素にGUIDを書き込み */
		virtual LONG WriteElement(const std::string&  name, const boost::uuids::uuid& value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, const boost::uuids::uuid& value) = 0;


		/** 要素に文字列を書き込み */
		virtual LONG AddElementString(const std::string& value) = 0;
		virtual LONG AddElementString(const std::wstring& value) = 0;
		/** 要素に文字列を書き込み */
		virtual LONG AddElementString(const char value[]) = 0;
		virtual LONG AddElementString(const WCHAR value[]) = 0;
		/** 要素に整数を書き込み */
		virtual LONG AddElementString(int value) = 0;
		/** 要素に符号無し整数を書き込み */
		virtual LONG AddElementString(unsigned long value) = 0;
		/** 要素に実数を書き込み */
		virtual LONG AddElementString(double value) = 0;
		/** 要素に論理値を書き込み */
		virtual LONG AddElementString(bool value) = 0;
		/** 要素にGUIDを書き込み */
		virtual LONG AddElementString(const boost::uuids::uuid& value) = 0;


		/** 要素に属性を追加 */
		virtual LONG AddAttribute(const std::string& id, const std::string& value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, const std::wstring& value) = 0;
		/** 要素に属性を追加(文字列) */
		virtual LONG AddAttribute(const std::string& id, const char value[]) = 0;
		virtual LONG AddAttribute(const std::wstring& id, const WCHAR value[]) = 0;
		/** 要素に属性を追加(整数32bit) */
		virtual LONG AddAttribute(const std::string& id, __int32 value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, __int32 value) = 0;
		/** 要素に属性を追加(整数64bit) */
		virtual LONG AddAttribute(const std::string& id, __int64 value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, __int64 value) = 0;
		/** 要素に属性を追加(倍精度実数) */
		virtual LONG AddAttribute(const std::string& id, double value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, double value) = 0;
		/** 要素に属性を追加(論理値) */
		virtual LONG AddAttribute(const std::string& id, bool value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, bool value) = 0;
		/** 要素に属性を追加する(GUID) */
		virtual LONG AddAttribute(const std::string& strAttrName, const boost::uuids::uuid& value) = 0;
		virtual LONG AddAttribute(const std::wstring& strAttrName, const boost::uuids::uuid& value) = 0;
	};
	typedef boost::shared_ptr<IXmlWriter> XMLWriterPtr;

	class XMLUTILITY_API IXmlReader
	{
	public:
		/** コンストラクタ */
		IXmlReader(){}
		/** デストラクタ */
		virtual ~IXmlReader(){}

	public:
		/** 次のノードを読み込む */
		virtual LONG Read(NodeType& nodeType) = 0;

		/** 現在の要素名を取得する */
		virtual std::string GetElementName() = 0;
		virtual std::wstring GetElementNameW() = 0;


		/** 値を文字列で取得する */
		virtual std::string ReadElementValueString() = 0;
		virtual std::wstring ReadElementValueWString() = 0;

		/** 値を10進整数で取得する */
		virtual __int32 ReadElementValueInt32() = 0;
		virtual __int64 ReadElementValueInt64() = 0;
		/** 値を16進整数で取得する */
		virtual __int32 ReadElementValueIntX32() = 0;
		virtual __int64 ReadElementValueIntX64() = 0;
		/** 値を実数で取得する */
		virtual double ReadElementValueDouble() = 0;
		/** 値を論理値で取得する */
		virtual bool ReadElementValueBool() = 0;
		/** 値をGUIDで取得する */
		virtual boost::uuids::uuid ReadElementValueGUID() = 0;


		/** 値を文字列配列の番号で取得する */
		virtual LONG ReadElementValueEnum(const std::string lpName[], LONG valueCount) = 0;
		virtual LONG ReadElementValueEnum(const std::wstring lpName[], LONG valueCount) = 0;

		/** 値を文字列配列の番号で取得する */
		virtual LONG ReadElementValueEnum(const std::vector<std::string>& lpName) = 0;
		virtual LONG ReadElementValueEnum(const std::vector<std::wstring>& lpName) = 0;

		/** 属性をリストで取得する */
		virtual std::map<std::string, std::string> ReadAttributeList() = 0;
		virtual std::map<std::wstring, std::wstring> ReadAttributeListW() = 0;
	};
	typedef boost::shared_ptr<IXmlReader> XMLReaderPtr;


	//==========================================
	// 関数定義
	//==========================================
	/** XML書き込みクラスを作成する */
	XMLUTILITY_API XMLWriterPtr CreateXMLWriter(const std::string& filePath);
	XMLUTILITY_API XMLWriterPtr CreateXMLWriter(const std::wstring& filePath);

	/** XML読み込みクラスを作成する */
	XMLUTILITY_API XMLReaderPtr CreateXMLReader(const std::string& filePath);
	XMLUTILITY_API XMLReaderPtr CreateXMLReader(const std::wstring& filePath);
}


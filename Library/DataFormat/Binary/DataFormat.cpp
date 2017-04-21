// StringArray.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"


#include"DataFormat.h"

#include<string>
#include<vector>
#include<list>
#include<set>
#include<map>

#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/xml_parser.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include"Library/Common/StringUtility/StringUtility.h"

#include"DataFormatItem.h"
#include"DataFormatClass.h"

namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	namespace
	{
		const std::vector<std::wstring> lpDataTypeID =
		{
			L"sbyte",		// DATA_TYPE_SBYTE,
			L"byte",		// DATA_TYPE_BYTE,
			L"short",		// DATA_TYPE_SHORT,
			L"ushort",		// DATA_TYPE_USHORT,
			L"int",			// DATA_TYPE_LONG,
			L"uint",		// DATA_TYPE_ULONG,
			L"longlong",	// DATA_TYPE_LONGLONG,
			L"ulonglong",	// DATA_TYPE_ULONGLONG,
			L"float",		// DATA_TYPE_FLOAT,
			L"double",		// DATA_TYPE_DOUBLE,
		};

		template<class Type>
		Type GetTreeValue(const boost::property_tree::ptree& pXmlTree, const std::string& id)
		{
			Type value = 0;
			if(boost::optional<Type> pValue = pXmlTree.get_optional<Type>(id))
			{
				value = pValue.get();
			}

			return value;
		}
		template<>
		std::wstring GetTreeValue<std::wstring>(const boost::property_tree::ptree& pXmlTree, const std::string& id)
		{
			std::wstring value;
			if(boost::optional<std::string> pValue = pXmlTree.get_optional<std::string>(id))
			{
				value = StringUtility::UTF8toUnicode(pValue.get());
			}

			return value;
		}
		template<>
		DataType GetTreeValue<DataType>(const boost::property_tree::ptree& pXmlTree, const std::string& id)
		{
			std::wstring value = GetTreeValue<std::wstring>(pXmlTree, id);

			for(U32 dataType=0; dataType<lpDataTypeID.size(); dataType++)
			{
				if(value == lpDataTypeID[dataType])
				{
					return (DataType)dataType;
				}
			}

			return DataType::DATA_TYPE_ULONG;
		}


		// アイテム配列を読み込む
		Format::CItem_data* ReadData(CDataFormat* pDataFormat, const boost::property_tree::ptree& pXmlTree);
		Format::CItem_items* ReadItems(CDataFormat* pDataFormat, const boost::property_tree::ptree& pXmlTree);

		Gravisbell::ErrorCode ReadItems(CDataFormat* pDataFormat, Format::CItem_items_base* pItems, const boost::property_tree::ptree& pXmlTree);
	}


	/** 文字列の配列を読み込むデータフォーマットを作成する */
	GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new CDataFormat(i_szName, i_szText, false);
	}
	/** 文字列の配列を読み込むデータフォーマットを作成する */
	GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormatFromXML(const wchar_t szXMLFilePath[])
	{
		using namespace StringUtility;

		// XMLファイルの読み込み
		boost::property_tree::ptree pXmlTree;
		boost::iostreams::file_descriptor_source fs(UnicodeToShiftjis(szXMLFilePath));
		boost::iostreams::stream<boost::iostreams::file_descriptor_source> fsstream(fs);
		try
		{
			boost::property_tree::read_xml(fsstream, pXmlTree);
		}
		catch(boost::exception& e)
		{
			e;
			return NULL;
		}

		CDataFormat* pDataFormat = NULL;
		try
		{
			// 名前
			std::wstring name = GetTreeValue<std::wstring>(pXmlTree, "DataFormat-binary.Name");
			// テキスト
			std::wstring text = GetTreeValue<std::wstring>(pXmlTree, "DataFormat-binary.Text");

			// バイトオーダーを取得する
			bool onReverseByteOrder = false;
			std::wstring byteOrder = GetTreeValue<std::wstring>(pXmlTree, "DataFormat-binary.<xmlattr>.byte-order");
			if(!byteOrder.empty())
			{
				// システム側のバイトオーダーを調べる
				ByteOrder systemByteOrder = ByteOrder::BYTEODER_BIG;
				{
					U16 value = 0x0A0B;
					U08* lpBuf = (U08*)&value;

					if(lpBuf[0] == 0x0A)
						systemByteOrder = BYTEODER_BIG;
					else
						systemByteOrder = BYTEODER_LITTLE;
				}

				// データ側のバイトオーダーを調べる
				ByteOrder dataByteOrder = ByteOrder::BYTEODER_BIG;
				if(byteOrder == L"big")
					dataByteOrder = BYTEODER_BIG;
				else if(byteOrder == L"little")
					dataByteOrder = BYTEODER_LITTLE;

				// 反転フラグ
				onReverseByteOrder = systemByteOrder != dataByteOrder;
			}


			// データフォーマットを作成
			pDataFormat = new CDataFormat(name.c_str(), text.c_str(), onReverseByteOrder);


			// アイテムを走査
			for(const boost::property_tree::ptree::value_type &it_root : pXmlTree.get_child("DataFormat-binary"))
			{
				if(it_root.first == "Structure")	// 構造
				{
					// 属性からカテゴリを取得
					std::wstring category = GetTreeValue<std::wstring>(it_root.second, "<xmlattr>.category");

					// X軸
					std::wstring var_x = GetTreeValue<std::wstring>(it_root.second, "x");
					// Y軸
					std::wstring var_y = GetTreeValue<std::wstring>(it_root.second, "y");
					// Z軸
					std::wstring var_z = GetTreeValue<std::wstring>(it_root.second, "z");
					// ch軸
					std::wstring var_ch = GetTreeValue<std::wstring>(it_root.second, "ch");

					// false
					F32 falseValue = GetTreeValue<F32>(it_root.second, "false");
					// true
					F32 trueValue  = GetTreeValue<F32>(it_root.second, "true");


					// フォーマットに構造を追加
					pDataFormat->AddDataInfo(category.c_str(), var_x.c_str(), var_y.c_str(), var_z.c_str(), var_ch.c_str(), falseValue, trueValue);
				}
				else if(it_root.first == "Data")
				{
					Format::CItem_data* pData = ReadData(pDataFormat, it_root.second);
					if(pData)
					{
						pDataFormat->AddDataFormat(pData);
					}
				}
			}
		}
		catch(boost::exception& e)
		{
			e;
			if(pDataFormat)
				delete pDataFormat;
			return NULL;
		}

		return pDataFormat;
	}


	namespace
	{
		// アイテム配列を読み込む
		Format::CItem_data* ReadData(CDataFormat* pDataFormat, const boost::property_tree::ptree& pTree)
		{
			Format::CItem_data* pItems = new Format::CItem_data(*pDataFormat);
			if(ReadItems(pDataFormat, pItems, pTree) != ErrorCode::ERROR_CODE_NONE)
			{
				delete pItems;
				return NULL;
			}

			return pItems;
		}
		Format::CItem_items* ReadItems(CDataFormat* pDataFormat, const boost::property_tree::ptree& pTree)
		{
			std::wstring id    = GetTreeValue<std::wstring>(pTree, "<xmlattr>.id");
			std::wstring count = GetTreeValue<std::wstring>(pTree, "<xmlattr>.count");
			
			Format::CItem_items* pItems = new Format::CItem_items(*pDataFormat, id, count);
			if(ReadItems(pDataFormat, pItems, pTree) != ErrorCode::ERROR_CODE_NONE)
			{
				delete pItems;
				return NULL;
			}

			return pItems;
		}
		Gravisbell::ErrorCode ReadItems(CDataFormat* pDataFormat, Format::CItem_items_base* pItems, const boost::property_tree::ptree& pTree)
		{
			using namespace StringUtility;

			for(const boost::property_tree::ptree::value_type &it_items : pTree.get_child(""))
			{
				if(it_items.first == "Signature")
				{
					// サイズ
					U32 size = GetTreeValue<U32>(it_items.second, "<xmlattr>.size");

					pItems->AddItem(new Format::CItem_signature(*pDataFormat, size, UTF8toUnicode(it_items.second.data())));
				}
				else if(it_items.first == "Variable")
				{
					// サイズ
					U32 size = GetTreeValue<U32>(it_items.second, "<xmlattr>.size");
					// 種別
					DataType dataType = GetTreeValue<DataType>(it_items.second, "<xmlattr>.type");
					// ID
					std::wstring id = GetTreeValue<std::wstring>(it_items.second, "<xmlattr>.id");

					pItems->AddItem(new Format::CItem_variable(*pDataFormat, size, dataType, id));
				}
				else if(it_items.first == "Float")
				{
					// カテゴリ
					std::wstring category = GetTreeValue<std::wstring>(it_items.second, "<xmlattr>.category");
					// サイズ
					U32 size = GetTreeValue<U32>(it_items.second, "<xmlattr>.size");
					// 種別
					DataType dataType = GetTreeValue<DataType>(it_items.second, "<xmlattr>.type");
					// 正規化
					std::wstring normalize = GetTreeValue<std::wstring>(it_items.second, "<xmlattr>.normalize");

					// no
					std::wstring no = GetTreeValue<std::wstring>(it_items.second, "no");
					std::wstring x  = GetTreeValue<std::wstring>(it_items.second, "x");
					std::wstring y  = GetTreeValue<std::wstring>(it_items.second, "y");
					std::wstring z  = GetTreeValue<std::wstring>(it_items.second, "z");
					std::wstring ch = GetTreeValue<std::wstring>(it_items.second, "ch");

					if(normalize == L"min-max")
					{
						std::wstring minValue = GetTreeValue<std::wstring>(it_items.second, "min");
						std::wstring maxValue = GetTreeValue<std::wstring>(it_items.second, "max");

						pItems->AddItem(new Format::CItem_float_normalize_min_max(*pDataFormat, category, size, dataType, no, x, y, z, ch, minValue, maxValue));
					}
					else
					{
						pItems->AddItem(new Format::CItem_float(*pDataFormat, category, size, dataType, no, x, y, z, ch));
					}
				}
				else if(it_items.first == "BoolArray")
				{
					// カテゴリ
					std::wstring category = GetTreeValue<std::wstring>(it_items.second, "<xmlattr>.category");
					// サイズ
					U32 size = GetTreeValue<U32>(it_items.second, "<xmlattr>.size");
					// 種別
					DataType dataType = GetTreeValue<DataType>(it_items.second, "<xmlattr>.type");

					// no
					std::wstring no = GetTreeValue<std::wstring>(it_items.second, "no");
					std::wstring x  = GetTreeValue<std::wstring>(it_items.second, "x");
					std::wstring y  = GetTreeValue<std::wstring>(it_items.second, "y");
					std::wstring z  = GetTreeValue<std::wstring>(it_items.second, "z");
					std::wstring ch = GetTreeValue<std::wstring>(it_items.second, "ch");

					pItems->AddItem(new Format::CItem_boolArray(*pDataFormat, category, size, dataType, no, x, y, z, ch));
				}
				else if(it_items.first == "Items")
				{
					Format::CItem_items* pAddItems = ReadItems(pDataFormat, it_items.second);
					if(pAddItems)
					{
						pItems->AddItem(pAddItems);
					}
				}
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	}

}	// Binary
}	// DataFormat
}	// Gravisbell



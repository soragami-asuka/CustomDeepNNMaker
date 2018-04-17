/*--------------------------------------------
 * FileName  : SignalArray2Value_FUNC.cpp
 * LayerName : 信号の配列から値へ変換
 * guid      : 97C8E5D3-AA0E-43AA-96C2-E7E434F104B8
 * 
 * Text      : 信号の配列から値へ変換する.
 *           : 最大値を取るCH番号を値に変換する.
 *           : 入力CH数＝分解能の整数倍である必要がある.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"SignalArray2Value_FUNC.hpp"


// {97C8E5D3-AA0E-43AA-96C2-E7E434F104B8}
static const Gravisbell::GUID g_guid(0x97c8e5d3, 0xaa0e, 0x43aa, 0x96, 0xc2, 0xe7, 0xe4, 0x34, 0xf1, 0x04, 0xb8);

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 



struct StringData
{
    std::wstring name;
    std::wstring text;
};

namespace DefaultLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = L"ja";

    /** Base */
    static const StringData g_baseData = 
    {
        L"信号の配列から値へ変換",
        L"信号の配列から値へ変換する.\n最大値を取るCH番号を値に変換する.\n入力CH数＝分解能の整数倍である必要がある."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"outputMinValue",
            {
                L"出力最小値",
                L"",
            }
        },
        {
            L"outputMaxValue",
            {
                L"出力最大値",
                L"",
            }
        },
        {
            L"resolution",
            {
                L"分解能",
                L"",
            }
        },
        {
            L"allocationType",
            {
                L"割り当て種別",
                L"CH番号→値に変換するための変換方法",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"allocationType",
            {
                {
                    L"max",
                    {
                        L"最大値",
                        L"CH内の最大値を出力する",
                    },
                },
                {
                    L"average",
                    {
                        L"平均",
                        L"CH番号とCHの値を掛け合わせた値の平均値を出力する(相加平均)",
                    },
                },
            }
        },
    };



    /** ItemData Runtiime Parameter <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Runtime = 
    {
    };


    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Runtime =
    {
    };


}


namespace CurrentLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = DefaultLanguage::g_languageCode;


    /** Base */
    static StringData g_baseData = DefaultLanguage::g_baseData;


    /** ItemData Layer Structure <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_LayerStructure = DefaultLanguage::g_lpItemData_LayerStructure;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure = DefaultLanguage::g_lpItemDataEnum_LayerStructure;


    /** ItemData Runtime <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Runtime;

    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Runtime;

}



/** Acquire the layer identification code.
  * @param  o_layerCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** Get version code.
  * @param  o_versionCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** Create a layer structure setting.
  * @return If successful, new configuration information.
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : 出力最小値
	  * ID   : outputMinValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"outputMinValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMinValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMinValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : 出力最大値
	  * ID   : outputMaxValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"outputMaxValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMaxValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMaxValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : 分解能
	  * ID   : resolution
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"resolution",
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].text.c_str(),
			2, 65535, 2));

	/** Name : 割り当て種別
	  * ID   : allocationType
	  * Text : CH番号→値に変換するための変換方法
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"allocationType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"allocationType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"allocationType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"max",
			L"最大値",
			L"CH内の最大値を出力する");
		// 1
		pItemEnum->AddEnumItem(
			L"average",
			L"平均",
			L"CH番号とCHの値を掛け合わせた値の平均値を出力する(相加平均)");

pItemEnum->SetDefaultItem(0);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	return pLayerConfig;
}

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a runtime parameters.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameter(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	return pLayerConfig;
}

/** Create runtime parameter from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateRuntimeParameter();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}



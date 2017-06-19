/*--------------------------------------------
 * FileName  : Convolution_FUNC.cpp
 * LayerName : 畳みこみニューラルネットワーク
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : 畳みこみニューラルネットワーク.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"Convolution_FUNC.hpp"


// {F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA}
static const Gravisbell::GUID g_guid(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa);

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
        L"畳みこみニューラルネットワーク",
        L"畳みこみニューラルネットワーク."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"FilterSize",
            {
                L"フィルタサイズ",
                L"畳みこみを行う入力信号数",
            }
        },
        {
            L"Output_Channel",
            {
                L"出力チャンネル数",
                L"出力されるチャンネルの数",
            }
        },
        {
            L"Stride",
            {
                L"フィルタ移動量",
                L"畳みこみごとに移動するフィルタの移動量",
            }
        },
        {
            L"Padding",
            {
                L"パディングサイズ",
                L"",
            }
        },
        {
            L"PaddingType",
            {
                L"パディング種別",
                L"パディングを行う際の方法設定",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"PaddingType",
            {
                {
                    L"zero",
                    {
                        L"ゼロパディング",
                        L"不足分を0で埋める",
                    },
                },
            }
        },
    };



    /** ItemData Learn <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Learn = 
    {
        {
            L"Optimizer",
            {
                L"最適化ルーチン",
                L"重み誤差を反映させるためのアルゴリズム.",
            }
        },
        {
            L"LearnCoeff",
            {
                L"学習係数",
                L"SGD,Momentumで使用.",
            }
        },
        {
            L"Momentum_alpha",
            {
                L"Momentum-α",
                L"Momentumで使用.t-1の値を反映する割合",
            }
        },
        {
            L"AdaDelta_rho",
            {
                L"AdaDelta-Ρ",
                L"AdaDeltaで使用.減衰率.高いほうが減衰しづらい.",
            }
        },
        {
            L"AdaDelta_epsilon",
            {
                L"AdaDelta-ε",
                L"AdaDeltaで使用.補助数.高いほど初期値が大きくなる.",
            }
        },
        {
            L"Adam_alpha",
            {
                L"Adam-α",
                L"Adamで使用.加算率.高いほうが更新されやすい.",
            }
        },
        {
            L"Adam_beta1",
            {
                L"Adam-β",
                L"Adamで使用.減衰率1.高いほうが減衰しづらい.",
            }
        },
        {
            L"Adam_beta2",
            {
                L"Adam-β",
                L"Adamで使用.減衰率2.高いほうが減衰しづらい.",
            }
        },
        {
            L"Adam_epsilon",
            {
                L"Adam-ε",
                L"AdaDeltaで使用.補助数.高いほど初期値が大きくなる.",
            }
        },
    };


    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn =
    {
        {
            L"Optimizer",
            {
                {
                    L"SGD",
                    {
                        L"SGD",
                        L"",
                    },
                },
                {
                    L"Momentum",
                    {
                        L"Momentum",
                        L"慣性付与",
                    },
                },
                {
                    L"AdaDelta",
                    {
                        L"AdaDelta",
                        L"",
                    },
                },
                {
                    L"Adam",
                    {
                        L"Adam",
                        L"",
                    },
                },
            }
        },
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


    /** ItemData Learn <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Learn;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Learn;

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
	/** Name : フィルタサイズ
	  * ID   : FilterSize
	  * Text : 畳みこみを行う入力信号数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"FilterSize",
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : 出力チャンネル数
	  * ID   : Output_Channel
	  * Text : 出力されるチャンネルの数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Output_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].text.c_str(),
			1, 65535, 1));

	/** Name : フィルタ移動量
	  * ID   : Stride
	  * Text : 畳みこみごとに移動するフィルタの移動量
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Stride",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : パディングサイズ
	  * ID   : Padding
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Padding",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

	/** Name : パディング種別
	  * ID   : PaddingType
	  * Text : パディングを行う際の方法設定
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"PaddingType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"zero",
			L"ゼロパディング",
			L"不足分を0で埋める");

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
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a learning setting.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSetting(void)
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
	/** Name : 最適化ルーチン
	  * ID   : Optimizer
	  * Text : 重み誤差を反映させるためのアルゴリズム.
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"Optimizer",
			CurrentLanguage::g_lpItemData_Learn[L"Optimizer"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Optimizer"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"SGD",
			L"SGD",
			L"");
		// 1
		pItemEnum->AddEnumItem(
			L"Momentum",
			L"Momentum",
			L"慣性付与");
		// 2
		pItemEnum->AddEnumItem(
			L"AdaDelta",
			L"AdaDelta",
			L"");
		// 3
		pItemEnum->AddEnumItem(
			L"Adam",
			L"Adam",
			L"");

pItemEnum->SetDefaultItem(0);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	/** Name : 学習係数
	  * ID   : LearnCoeff
	  * Text : SGD,Momentumで使用.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"LearnCoeff",
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].text.c_str(),
			0.0000000000000000f, 1000.0000000000000000f, 1.0000000000000000f));

	/** Name : Momentum-α
	  * ID   : Momentum_alpha
	  * Text : Momentumで使用.t-1の値を反映する割合
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Momentum_alpha",
			CurrentLanguage::g_lpItemData_Learn[L"Momentum_alpha"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Momentum_alpha"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.8999999761581421f));

	/** Name : AdaDelta-Ρ
	  * ID   : AdaDelta_rho
	  * Text : AdaDeltaで使用.減衰率.高いほうが減衰しづらい.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"AdaDelta_rho",
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_rho"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_rho"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.9499999880790710f));

	/** Name : AdaDelta-ε
	  * ID   : AdaDelta_epsilon
	  * Text : AdaDeltaで使用.補助数.高いほど初期値が大きくなる.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"AdaDelta_epsilon",
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_epsilon"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_epsilon"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.0000009999999975f));

	/** Name : Adam-α
	  * ID   : Adam_alpha
	  * Text : Adamで使用.加算率.高いほうが更新されやすい.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_alpha",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_alpha"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_alpha"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.0010000000474975f));

	/** Name : Adam-β
	  * ID   : Adam_beta1
	  * Text : Adamで使用.減衰率1.高いほうが減衰しづらい.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_beta1",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta1"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta1"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.8999999761581421f));

	/** Name : Adam-β
	  * ID   : Adam_beta2
	  * Text : Adamで使用.減衰率2.高いほうが減衰しづらい.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_beta2",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta2"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta2"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.9990000128746033f));

	/** Name : Adam-ε
	  * ID   : Adam_epsilon
	  * Text : AdaDeltaで使用.補助数.高いほど初期値が大きくなる.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_epsilon",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_epsilon"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_epsilon"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.0000000099999999f));

	return pLayerConfig;
}

/** Create learning settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLearningSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}



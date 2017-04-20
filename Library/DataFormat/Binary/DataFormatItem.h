//==========================================
// フォーマットを定義するためのアイテム
//==========================================
#ifndef __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_ITEM_H__
#define __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_ITEM_H__


#include<string>
#include<vector>
#include<list>
#include<set>
#include<map>

#include"DataFormat.h"


namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	class CDataFormat;

	enum DataType
	{
		DATA_TYPE_SBYTE,
		DATA_TYPE_BYTE,
		DATA_TYPE_SHORT,
		DATA_TYPE_USHORT,
		DATA_TYPE_LONG,
		DATA_TYPE_ULONG,
		DATA_TYPE_LONGLONG,
		DATA_TYPE_ULONGLONG,
		DATA_TYPE_FLOAT,
		DATA_TYPE_DOUBLE,
	};


	namespace Format
	{
		namespace Data
		{
			class CDataItem_base
			{
			protected:
				CDataFormat& m_dataFormat;

			public:
				/** コンストラクタ */
				CDataItem_base(CDataFormat& i_dataFormat)
					:	m_dataFormat	(i_dataFormat)
				{
				}
				/** デストラクタ */
				virtual ~CDataItem_base(){}

			public:
				/** バイナリデータを読み込む.
					@param	i_lpBuf			バイナリ先頭アドレス.
					@param	i_byteCount		読込可能なバイト数.
					@param	i_dataNo		データ番号.
					@param	i_lpLocalValue	ローカル変数.
					@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
				virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount, U32 i_dataNo) = 0;
			};
			class CDataItem_float : public CDataItem_base
			{
			private:

			protected:
				std::wstring m_category;

				U32 m_size;
				DataType m_dataType;

				std::wstring var_x;
				std::wstring var_y;
				std::wstring var_z;
				std::wstring var_ch;

			public:
				/** コンストラクタ */
				CDataItem_float(
					CDataFormat& i_dataFormat,
					const std::wstring& i_category,
					U32 i_size, DataType i_dataType,
					const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch);
				/** デストラクタ */
				virtual ~CDataItem_float();

			public:
				/** バイナリデータを読み込む.
					@param	i_lpBuf			バイナリ先頭アドレス.
					@param	i_byteCount		読込可能なバイト数.
					@param	i_dataNo		データ番号.
					@param	i_lpLocalValue	ローカル変数.
					@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
				virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount, U32 i_dataNo);
			};
			class CDataItem_float_normalize_min_max : public CDataItem_base
			{
			protected:
				const std::wstring& var_min;
				const std::wstring& var_max;

			public:
				/** コンストラクタ */
				CDataItem_float_normalize_min_max(
					CDataFormat& i_dataFormat,
					U32 i_size, DataType i_dataType,
					const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch,
					const std::wstring& i_var_min, const std::wstring& i_var_max);
				/** デストラクタ */
				virtual ~CDataItem_float_normalize_min_max();

			public:
				/** バイナリデータを読み込む.
					@param	i_lpBuf			バイナリ先頭アドレス.
					@param	i_byteCount		読込可能なバイト数.
					@param	i_dataNo		データ番号.
					@param	i_lpLocalValue	ローカル変数.
					@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
				virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount, U32 i_dataNo);
			};
			class CDataItem_int : public CDataItem_base
			{
			protected:
				U32 m_size;
				DataType m_dataType;

				std::wstring var_x;
				std::wstring var_y;
				std::wstring var_z;
				std::wstring var_ch;

			public:
				/** コンストラクタ */
				CDataItem_int(
					CDataFormat& i_dataFormat,
					U32 i_size, DataType i_dataType,
					const std::wstring& i_var_x, const std::wstring& i_var_y, const std::wstring& i_var_z, const std::wstring& i_var_ch);
				/** デストラクタ */
				virtual ~CDataItem_int();

			public:
				/** バイナリデータを読み込む.
					@param	i_lpBuf			バイナリ先頭アドレス.
					@param	i_byteCount		読込可能なバイト数.
					@param	i_dataNo		データ番号.
					@param	i_lpLocalValue	ローカル変数.
					@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
				virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount, U32 i_dataNo);
			};
			class CDataItem_items : public CDataItem_base
			{
			protected:
				std::wstring m_id;
				std::wstring m_count;

			public:
				/** コンストラクタ */
				CDataItem_items(CDataFormat& i_dataFormat, U32 i_id, U32 i_count);
				/** デストラクタ */
				virtual ~CDataItem_items();

			public:
				/** バイナリデータを読み込む.
					@param	i_lpBuf			バイナリ先頭アドレス.
					@param	i_byteCount		読込可能なバイト数.
					@param	i_dataNo		データ番号.
					@param	i_lpLocalValue	ローカル変数.
					@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
				virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount, U32 i_dataNo);
			};
		}

		class CItem_base
		{
		protected:
			CDataFormat& dataFormat;

		public:
			/** コンストラクタ */
			CItem_base(CDataFormat& i_dataFormat)
				:	dataFormat	(i_dataFormat)
			{
			}
			/** デストラクタ */
			virtual ~CItem_base(){}

		public:
			/** バイナリデータを読み込む.
				@param	i_lpBuf			バイナリ先頭アドレス.
				@param	i_byteCount		読込可能なバイト数.
				@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
			virtual S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount) = 0;
		};
		class CItem_blank : public CItem_base
		{
		private:
			U32 m_size;

		public:
			/** コンストラクタ */
			CItem_blank(CDataFormat& i_dataFormat, U32 i_size);
			/** デストラクタ */
			virtual ~CItem_blank();
			
		public:
			/** バイナリデータを読み込む.
				@param	i_lpBuf			バイナリ先頭アドレス.
				@param	i_byteCount		読込可能なバイト数.
				@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
			S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_signature : public CItem_base
		{
		private:
			U32 m_size;

			std::vector<BYTE> m_lpBuf;	/**< シグネチャとして使用するバッファ.読み込んだ情報とこのバッファとの比較を行う */

		public:
			/** コンストラクタ */
			CItem_signature(CDataFormat& i_dataFormat, U32 i_size, const std::wstring& i_buf);
			/** デストラクタ */
			virtual ~CItem_signature();

		public:
			/** バイナリデータを読み込む.
				@param	i_lpBuf			バイナリ先頭アドレス.
				@param	i_byteCount		読込可能なバイト数.
				@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
			S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_variable : public CItem_base
		{
		private:
			U32 m_size;
			DataType m_dataType;
			std::wstring m_id;	/**< 変数ID */

		public:
			/** コンストラクタ */
			CItem_variable(CDataFormat& i_dataFormat, U32 i_size, DataType i_dataType, const std::wstring& i_id);
			/** デストラクタ */
			virtual ~CItem_variable();

		public:
			/** バイナリデータを読み込む.
				@param	i_lpBuf			バイナリ先頭アドレス.
				@param	i_byteCount		読込可能なバイト数.
				@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
			S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
		class CItem_data : public CItem_base
		{
		private:
			std::wstring m_count;	/**< ループ回数. 文字列の場合は変数名. */

			std::list<Data::CDataItem_base*> lpItem;

		public:
			/** コンストラクタ */
			CItem_data(CDataFormat& dataFormat, const std::wstring& i_count);
			/** デストラクタ */
			virtual ~CItem_data();

		public:
			/** アイテムを追加する */
			Gravisbell::ErrorCode AddItem(Data::CDataItem_base* pItem);

		public:
			/** バイナリデータを読み込む.
				@param	i_lpBuf		バイナリ先頭アドレス.
				@param	i_byteCount	読込可能なバイト数.
				@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
			S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
		};
	}

	
}	// Binary
}	// DataFormat
}	// Gravisbell


#endif	// __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_ITEM_H__
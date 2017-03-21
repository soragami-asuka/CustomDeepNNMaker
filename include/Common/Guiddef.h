//=======================================
// àÍî ê›íË
//=======================================
#ifndef __GRAVISBELL_GUIDDEF_H__
#define __GRAVISBELL_GUIDDEF_H__

#include"Common.h"

namespace Gravisbell {

	typedef struct _GUID {

		union
		{
			struct
			{
				U32 Data1;
				U16 Data2;
				U16 Data3;
				U08 Data4[ 8 ];
			};
			U08 data[16];
		};

		_GUID()
		{
			for(U32 i=0; i<16; i++)
				this->data[i] = 0;
		}
		_GUID(U32 i_Data1, U16 i_Data2, U16 i_Data3, U08 i_Data4_0, U08 i_Data4_1, U08 i_Data4_2, U08 i_Data4_3, U08 i_Data4_4, U08 i_Data4_5, U08 i_Data4_6, U08 i_Data4_7)
		{
			this->Data1 = i_Data1;
			this->Data2 = i_Data2;
			this->Data3 = i_Data3;
			this->Data4[0] = i_Data4_0;
			this->Data4[1] = i_Data4_1;
			this->Data4[2] = i_Data4_2;
			this->Data4[3] = i_Data4_3;
			this->Data4[4] = i_Data4_4;
			this->Data4[5] = i_Data4_5;
			this->Data4[6] = i_Data4_6;
			this->Data4[7] = i_Data4_7;
		}
		_GUID(const struct _GUID& i_value)
		{
			for(U32 i=0; i<16; i++)
				this->data[i] = i_value.data[i];
		}
		_GUID(const U08 i_data[])
		{
			for(U32 i=0; i<16; i++)
				this->data[i] = i_data[i];
		}


		bool operator==(const struct _GUID& value)const
		{
			for(U32 i=0; i<16; i++)
			{
				if(data[i] != value.data[i])
					return false;
			}
			return true;
		}
		bool operator!=(const struct _GUID& value)const
		{
			return !(*this == value);
		}
		bool operator<(const struct _GUID& value)const
		{
			for(U32 i=0; i<16; i++)
			{
				if(data[i] < value.data[i])
					return true;
				if(data[i] > value.data[i])
					return false;
			}
			return false;
		}
		const struct _GUID& operator=(const struct _GUID& value)
		{
			for(U32 i=0; i<16; i++)
			{
				this->data[i] = value.data[i];
			}

			return *this;
		}
		const struct _GUID& operator=(const U08 i_data[])
		{
			for(U32 i=0; i<16; i++)
			{
				this->data[i] = i_data[i];
			}

			return *this;
		}

	} GUID;


}	// Gravisbell


#endif // __GRAVISBELL_COMMON_H__
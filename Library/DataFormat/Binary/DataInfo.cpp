//====================================
// �f�[�^�{�̂̒�`
//====================================
#include"stdafx.h"

#include"DataFormatClass.h"


namespace Gravisbell {
namespace DataFormat {
namespace Binary {


	/** �R���X�g���N�^ */
	DataInfo::DataInfo()
	{
	}
	/** �f�X�g���N�^ */
	DataInfo::~DataInfo()
	{
	}

	/** �f�[�^�����擾 */
	U32 DataInfo::GetDataCount()const
	{
		return (U32)this->lpData.size();
	}

}	// Binary
}	// DataFormat
}	// Gravisbell